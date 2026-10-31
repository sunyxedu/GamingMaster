"""Pressure-first prompt binding with a checked tactical executor."""
from __future__ import annotations
import itertools
import json
import re
import time
from concurrent.futures import TimeoutError as FutureTimeout
from metaharness.core.contract import BudgetExceeded, ModelUnavailable

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
TRAPS = {'Mirror Force', 'Trap Hole', 'Compulsory Evacuation Device', 'Call of the Haunted'}
QUICK = {'Book of Moon', 'Rush Recklessly', 'Mystical Space Typhoon'}

def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), default=str)

def num(x, default=0):
    try: return int(x)
    except (ValueError, TypeError): return default

def players(o):
    ps = o.get('players') or []
    me = next((p for p in ps if p.get('role') == 'you' or p.get('player') == o.get('agent_seat')), {})
    en = next((p for p in ps if p is not me), {})
    return me, en

def card_info(c, o):
    if not isinstance(c, dict): return {}
    c = c.get('card', c)
    out = dict(o.get('card_catalog', {}).get(str(c.get('code')), {}))
    me, en = players(o)
    for p in (me, en):
        for zone in ('hand', 'monsters', 'spells_traps', 'graveyard', 'banished'):
            for actual in p.get(zone, []):
                if all(actual.get(k) == c.get(k) for k in ('player', 'zone', 'slot')) and c.get('slot') is not None:
                    out.update(actual)
    out.update(c)
    return out

def attack(c): return num(c.get('attack'))
def defense(c): return num(c.get('defense'))
def faceup(c): return 'down' not in str(c.get('position', 'face-down')).lower()
def atkpos(c): return 'attack' in str(c.get('position', '')).lower()
def value(c):
    return 800 + max(attack(c), defense(c)*0.65) + (450 if c.get('name') in FLIPS and not faceup(c) else 0)

def constraints(p):
    d = dict(p)
    for key in ('constraint', 'constraints'):
        if isinstance(p.get(key), dict): d.update(p[key])
    lo = num(d.get('min', d.get('minimum', d.get('min_count', 0))))
    hi = num(d.get('max', d.get('maximum', d.get('max_count', len(p.get('candidates', []))))), len(p.get('candidates', [])))
    tribute = 'tribute' in dumps({k:v for k,v in d.items() if k != 'candidates'}).lower()
    return lo, hi, tribute

def candidate_ids(p):
    return [num(c.get('index'), i) if isinstance(c, dict) else i for i,c in enumerate(p.get('candidates', []))]

def legal_subset(p, indices):
    ids = candidate_ids(p)
    if len(set(indices)) != len(indices) or any(i not in ids for i in indices): return False
    lo, hi, tribute = constraints(p)
    if len(indices) > hi: return False
    if tribute:
        cs = dict(zip(ids, p.get('candidates', [])))
        weights = dict(zip(ids, p.get('weights', [1]*len(ids))))
        return sum(num(cs[i].get('weight', cs[i].get('tribute_value', weights.get(i, 1))), 1) for i in indices) >= lo
    return len(indices) >= lo

def parse_action(text, o):
    if not isinstance(text, str): return None
    p = o.get('prompt') or {}
    legal = {c.get('action') for c in p.get('choices', [])}
    for m in re.finditer(r'\bchoose\s+[^\s"`]+\s+\d+\b', text):
        if m.group() in legal: return m.group()
    token = str(p.get('token', ''))
    for m in re.finditer(r'\b(select|order)\s+' + re.escape(token) + r'\s+(-|\d+(?:\s*,\s*\d+)*)', text):
        verb, raw = m.groups()
        indices = [] if raw == '-' else [int(x.strip()) for x in raw.split(',')]
        ids = candidate_ids(p)
        if verb == 'order':
            valid = sorted(indices) == sorted(ids)
        else: valid = legal_subset(p, indices)
        if valid: return verb + ' ' + token + ' ' + (','.join(map(str, indices)) if indices else '-')
    return None

def compact(o):
    out = dict(o)
    # Keep exact effect text, omit normal monster flavor and redundant base metadata.
    out['card_catalog'] = {k:{f:v for f,v in c.items() if f in ('name','attack','defense','level','text','type') and (f != 'text' or num(c.get('type')) != 17)} for k,c in o.get('card_catalog', {}).items()}
    out['recent_events'] = o.get('recent_events', [])[-14:]
    return out

class Policy:
    def __init__(self):
        self.source = ''
        self.attacker = None
        self.rejected = set()
        self.last_token = None

    def target_score(self, c, o, p):
        c = card_info(c, o)
        own = c.get('player') == o.get('agent_seat')
        source = self.source
        hint = str(p.get('hint', '')).lower()
        if 'tribute' in str(p.get('kind','')).lower() or 'tribute' in hint:
            return -value(c)
        if 'discard' in hint:
            return -value(c) - (800 if c.get('name') in TRAPS|QUICK else 0) + (1000 if num(c.get('level')) >= 7 else 0)
        if source in ('Monster Reborn','Call of the Haunted') or 'special summon' in hint:
            return attack(c) + (300 if c.get('name') in FLIPS else 0)
        if source == 'Magician of Faith':
            return {'Monster Reborn':4000,'Dark Hole':3500,'Fissure':3200,'Book of Moon':2800,'Mystical Space Typhoon':2000,'Rush Recklessly':1500}.get(c.get('name'), 0)
        if source == 'Rush Recklessly':
            return (5000 if own else -5000) + attack(c)
        if source == 'Book of Moon':
            return (-4000 if own else 4000) + attack(c)-defense(c)
        if source in ('Man-Eater Bug','Penguin Soldier','Compulsory Evacuation Device','Trap Hole','Mystical Space Typhoon','Fissure'):
            return (-5000 if own else 5000) + value(c)
        if 'attack' in hint or 'battle' in str(p.get('kind','')):
            a = self.attacker or {}
            if own: return -10000
            if not c.get('name'): return 250 if attack(a) >= 2000 else -500
            stat = attack(c) if atkpos(c) else defense(c)
            delta = attack(a)-stat
            return (2000 + value(c)*0.4 + max(0,delta) if delta > 0 else -2000 + delta)
        return (-2000 if own else 2000) + value(c)

    def select(self, o):
        p = o.get('prompt') or {}
        cs = p.get('candidates') or []
        ids = candidate_ids(p)
        token = str(p.get('token', ''))
        if 'order' in str(p.get('kind','')).lower():
            return 'order '+token+' '+','.join(map(str, ids))
        ranked = sorted(zip(ids,cs), key=lambda x:self.target_score(x[1],o,p), reverse=True)
        lo, hi, tribute = constraints(p)
        # Enumerate only tiny pool subsets. For large target lists sort and select.
        best, best_v = None, -float('inf')
        for k in range(0, min(hi, len(cs), 8)+1):
            if not tribute and k < lo: continue
            combos = itertools.combinations(ranked, k) if len(cs) <= 12 else [tuple(ranked[:k])]
            for comb in combos:
                chosen = [i for i,c in comb]
                if not legal_subset(p,chosen): continue
                sc = sum(self.target_score(c,o,p) for i,c in comb)
                if tribute: sc -= 10000*k
                # Avoid gratuitous optional own-targeting.
                if sc > best_v: best,best_v=chosen,sc
        if best is None: best = ids[:max(0, min(lo, hi))]
        return 'select '+token+' '+(','.join(map(str,best)) if best else '-')

    def pressure(self, choice, o):
        """A cheap win/survival pass that outranks generic card valuation."""
        me, en = players(o)
        lp = num(en.get('lp'), 8000)
        mylp = num(me.get('lp'), 8000)
        label = str(choice.get('label','')).lower()
        card = card_info(choice.get('card') or {}, o)
        a = attack(card)
        mine = [x for x in me.get('monsters',[]) if faceup(x) and atkpos(x)]
        opp = list(en.get('monsters',[]))
        # Attack choices are scored by the damage they can actually create now.
        if 'attack' in label and 'position' not in label:
            if not opp:
                dmg = a
            else:
                dmg = 0
                for t in opp:
                    stat = attack(t) if atkpos(t) else defense(t)
                    if a > stat: dmg = max(dmg, a-stat if atkpos(t) else 0)
            if dmg >= lp: return 200000
            if dmg > 0: return 9000 + dmg
            # Do not suicide a useful attacker into a larger visible body.
            return -5000
        # Entering battle is valuable when it exposes a lethal line.
        if 'battle phase' in label:
            total = 0
            for m in mine:
                ma = attack(m)
                if not opp: total += ma
                else:
                    wins = [ma-(attack(t) if atkpos(t) else defense(t)) for t in opp if ma > (attack(t) if atkpos(t) else defense(t))]
                    total += max(wins, default=0)
            if total >= lp: return 12000
            return 700 + total
        # A summon that raises available damage is preferred when close to zero.
        if 'normal summon' in label or 'special summon' in label:
            if a and lp <= 3200: return 2500 + a
            if mylp <= 1800 and a < 1600: return -1200
        # When a single removal opens lethal, spend it before another summon.
        if 'activat' in label and card.get('name') in ('Fissure','Dark Hole','Monster Reborn','Compulsory Evacuation Device','Penguin Soldier','Man-Eater Bug'):
            if opp and lp <= 4200: return 1800
        if ('end turn' in label or 'end phase' in label) and lp <= 3000 and mine:
            return -2500
        return 0

    def score(self, choice, o):
        urgent = self.pressure(choice, o)
        me,en = players(o)
        own,opp = me.get('monsters',[]), en.get('monsters',[])
        strongest = max((attack(c) for c in opp if faceup(c)), default=0)
        mine = max((attack(c) for c in own if faceup(c)), default=0)
        card = card_info(choice.get('card') or {}, o)
        name = card.get('name','')
        label = str(choice.get('label','')).lower()
        p=o.get('prompt') or {}
        kind=str(p.get('kind','')).lower()
        a,d=attack(card),defense(card)
        turnmine=o.get('turn_player')==o.get('agent_seat')
        chain=o.get('chain') or []
        if urgent >= 10000: return urgent
        if urgent <= -4000: return urgent
        if 'shuffle' in label: return -100000
        if 'cancel' in label or 'finish' in label or 'decline' in label or 'no chain' in label or label in ('no','pass','decline','do not activate'): return 0
        if label in ('yes','activate effect') and not card:
            return 500 if self.source in FLIPS else 5
        if 'normal summon' in label or 'special summon' in label:
            if num(card.get('level')) > 4:
                count=1 if num(card.get('level'))<=6 else 2
                cost=sum(sorted(value(c) for c in own)[:count])
                return 1200+a-cost+(1500 if a>strongest>=mine else 0)
            return 1000+a+(600 if a>strongest else -500) - (2500 if name in FLIPS else 0)
        if 'set monster' in label:
            if name=='Man-Eater Bug': return 3200 if opp else 1800
            if name=='Penguin Soldier': return 3100 if len(opp)>=2 else 2300
            if name=='Magician of Faith':
                spells=any(num(c.get('type'))&2 for c in me.get('graveyard',[]))
                return 2400 if spells else 100
            return 1200+d + (500 if opp and strongest>=mine and d>=strongest else -700)
        if 'flip summon' in label:
            return 3600 if name in FLIPS and opp else (2000+a if a>strongest else -200)
        if 'change position' in label:
            if atkpos(card): return 1500 if a<strongest and d>a else -400
            return 1600 if a>strongest or (not opp and a>=1300) else -400
        if 'set spell/trap' in label:
            if name in TRAPS: return 1500
            if name in QUICK: return 800 if o.get('phase')!='Main 1' or not opp else 150
            return -150
        if 'activat' in label:
            if name in FLIPS: return 5000
            if name=='Trap Hole': return 4500 if opp else -1000
            if name=='Mirror Force': return 5000 if opp else -1000
            if name=='Fissure': return 3900 if any(faceup(c) for c in opp) else -1000
            if name=='Dark Hole':
                delta=sum(value(c) for c in opp)-sum(value(c) for c in own)
                return 4000+delta if delta>800 or (strongest>mine and len(opp)>len(own)) else -1000
            if name in ('Monster Reborn','Call of the Haunted'):
                gy=me.get('graveyard',[])+(en.get('graveyard',[]) if name=='Monster Reborn' else [])
                best=max((attack(c) for c in gy if num(c.get('type'))&1), default=0)
                return 2500+best if best>=1700 and (turnmine or len(own)==0 or chain) else -500
            if name=='Mystical Space Typhoon':
                back=en.get('spells_traps',[])
                return 3700 if back and not chain else (-200 if chain else -1000)
            if name=='Compulsory Evacuation Device':
                return 3300 if opp and (strongest>mine or (not turnmine and 'battle' in str(o.get('phase','')).lower()) or any(num(c.get('level'))>4 for c in opp)) else -300
            if name=='Book of Moon':
                weak=any(faceup(c) and attack(c)>=mine>defense(c) for c in opp)
                return 3400 if turnmine and weak and not chain else -200
            if name=='Rush Recklessly':
                # Damage windows are where the temporary boost yields a real trade.
                phase=str(o.get('phase','')).lower()
                if 'damage' in phase and mine and strongest and 0<=strongest-mine<700: return 4200
                return -100
            return -50
        if 'attack' in label:
            if 'position' in label:
                return 500 + urgent if (not opp or a>=strongest or a>=1800) else 0
            if not card: return 500 + urgent
            if not opp: return 3000+a
            outcomes=[]
            for t in opp:
                if not t.get('name'): outcomes.append(1000 if a>=2000 else 100)
                else:
                    diff=a-(attack(t) if atkpos(t) else defense(t))
                    outcomes.append(2200+diff if diff>0 else (500 if diff==0 and atkpos(t) else -1000+diff))
            return max(outcomes,default=0) + urgent
        if 'defense' in label and 'position' in label: return 300
        if 'battle phase' in label: return 250 + urgent
        if 'main phase 2' in label or 'main 2' in label: return 100
        if 'end turn' in label or 'end phase' in label: return 50 + urgent
        if card: return self.target_score(card,o,p)
        return 0

    def choose(self,o):
        p=o.get('prompt') or {}
        token=p.get('token')
        if token!=self.last_token: self.rejected=set(); self.last_token=token
        choices=p.get('choices') or []
        # Subset prompts may ALSO offer a cancel choice. The candidates are the
        # actual target decisions; prioritizing choices alone cancels forever.
        if p.get('mode') == 'subset' or (p.get('candidates') and p.get('grammar','').startswith('select')):
            return self.select(o)
        if choices:
            available=[c for c in choices if c.get('action') not in self.rejected] or choices
            return max(available,key=lambda c:self.score(c,o)).get('action','noop')
        if 'candidates' in p: return self.select(o)
        return None

    def record(self,action,o):
        p=o.get('prompt') or {}
        choice=next((c for c in p.get('choices',[]) if c.get('action')==action),{})
        label=choice.get('label','').lower()
        if 'activat' in label:
            self.source=card_info(choice.get('card') or {},o).get('name',self.source)
        if 'attack' in label and 'position' not in label:
            self.attacker=card_info(choice.get('card') or {},o)
        # Native hints may identify a flip trigger independently of our last activation.
        hint=str(p.get('hint',''))
        match=re.search(r'effect (\d+)',hint)
        if match:
            effect=o.get('card_catalog',{}).get(match.group(1),{})
            if num(effect.get('type')) != 17: self.source=effect.get('name',self.source)
        for name in FLIPS|TRAPS|QUICK|{'Monster Reborn','Fissure','Dark Hole'}:
            if name.lower() in hint.lower(): self.source=name

class Harness:
    def __init__(self,ctx): self.ctx=ctx

    def run_episode(self,game):
        ctx=self.ctx
        obs=game.observe()
        policy=Policy()
        calls={}
        consulted=set()
        failure=None
        note=''
        pending=None
        current_round=None
        last_illegal=0
        turn_start_lp=None
        visits={}
        while not ctx.done and not game.done:
            if not isinstance(obs,dict): obs={}
            p=obs.get('prompt') or {}
            rnd=obs.get('round',0)
            if rnd!=current_round:
                current_round=rnd; policy=Policy(); note=''; failure=None
            progress=ctx.progress()
            if progress.get('illegal_actions',0)>last_illegal or progress.get('since_obs_change',0)>=2:
                failure='Action rejected or observation stalled: '+dumps(progress)
            last_illegal=progress.get('illegal_actions',0)
            policy.record('',obs)
            fallback=policy.choose(obs)
            # Tokens and event logs change even when an attack/cancel cycle
            # returns to the same physical state. Detect the semantic cycle.
            state_key=dumps([rnd,obs.get('turn'),obs.get('phase'),obs.get('players'),obs.get('chain'),p.get('kind')])
            visits[state_key]=visits.get(state_key,0)+1
            if visits[state_key]>=3:
                failure='Repeated physical board and decision kind despite new prompt tokens; previous line made no progress.'
                exits=[c for c in p.get('choices',[]) if any(x in c.get('label','').lower() for x in ('end turn','main phase 2','decline','cancel'))]
                if exits: fallback=exits[0].get('action')
            key=(rnd,obs.get('turn'))
            kind=str(p.get('kind',''))
            me,en=players(obs)
            ownturn=obs.get('turn_player')==obs.get('agent_seat')
            strategic=ownturn and kind=='main_command' and key not in consulted
            # Reserve advice across all duels rather than exhausting it in duel 1.
            ask=bool(failure) or (strategic and calls.get(rnd,0)<4)
            generic=not p
            if generic and not consulted: ask=True
            budget=ctx.budget.remaining()
            action=None
            if pending is not None and pending.done():
                try: pending.result()
                except Exception: pass
                pending=None
            if ask and pending is None and budget.get('system2_calls',0)>(0 if generic else 1) and (generic or budget.get('wall_s',0)>180):
                consulted.add(key); calls[rnd]=calls.get(rnd,0)+1
                prompt=(
                    'Play this Yu-Gi-Oh fixed beginner pool duel to win. Native choices are authoritative. '
                    'You advise a local tactical executor once per own turn. Choose the most important immediate action. '
                    'First check whether the current legal attack or removal creates lethal this turn; close low LP games before farming card value. Preserve survival when behind. Prioritize strong attackers, flip value and useful removal. '
                    'Do not activate interaction just because legal. MST does not negate a normal spell/trap. '
                    'Avoid tributing two productive attackers for Blue-Eyes. Set defensive traps and flip monsters. '
                    'Return JSON {"action":"exact current action", "note":"short plan and main danger"}. '
                    'For subset prompts obey the displayed count/weight constraint; order contains all indices. '
                    'No tools. Keep answer under 100 words.\n'
                    'PREVIOUS PLAN: '+note+'\nFAILURE EVIDENCE: '+str(failure)+'\n'
                    'LOCAL PROPOSAL: '+str(fallback)+'\nCURRENT OBSERVATION:\n'+dumps(compact(obs)))
                if generic: prompt+='\n'+game.describe()+'\nAction names: '+dumps(list(game.action_spec.names))
                try:
                    # A bounded wait also covers queue delays before the model's own timeout.
                    pending=ctx.spawn(lambda prompt=prompt:ctx.think(prompt,session_id=None,timeout_s=140.0))
                    res=pending.result(timeout=160.0)
                    pending=None
                    action=parse_action(res.text,obs)
                    try:
                        match=re.search(r'\{.*\}',res.text,re.S)
                        data=json.loads(match.group()) if match else {}
                        note=str(data.get('note',''))[:600]
                    except Exception: note=res.text[:400]
                    if generic:
                        names=list(game.action_spec.names)
                        action=next((a for a in names if a in res.text),names[0] if names else 'noop')
                    failure=None
                except (BudgetExceeded,ModelUnavailable):
                    if ctx.done: return
                except (TimeoutError,RuntimeError,FutureTimeout) as exc:
                    failure='Advice unavailable: '+str(exc)[:160]
                except Exception as exc:
                    failure='Unusable model response: '+str(exc)[:160]
            if ctx.done: return
            action=action or fallback
            if action is None:
                names=list(game.action_spec.names)
                action=names[0] if names else 'noop'
            policy.record(action,obs)
            old_token=p.get('token')
            before=obs
            try:
                nxt=ctx.act(action)
                obs=nxt if isinstance(nxt,dict) else ctx.observe()
            except (BudgetExceeded,ModelUnavailable): return
            except Exception as exc:
                policy.rejected.add(action)
                failure='Action exception: '+str(exc)[:200]
                obs=ctx.observe()
                continue
            after=ctx.progress()
            err=obs.get('last_action_error') if isinstance(obs,dict) else None
            new_token=(obs.get('prompt') or {}).get('token')
            if err or (old_token is not None and new_token==old_token):
                policy.rejected.add(action)
                failure=dumps({'action':action,'error':err,'progress':after})
            elif after.get('illegal_actions',0)<=last_illegal:
                failure=None
            # A material reversal merits advice at the next own main decision.
            bm,be=players(before); am,ae=players(obs)
            if num(bm.get('lp'))-num(am.get('lp'))>=2000 or len(bm.get('monsters',[]))-len(am.get('monsters',[]))>=2:
                consulted.discard((obs.get('round'),obs.get('turn')))
                failure='Board reversal: '+dumps({'own_lp_before':bm.get('lp'),'own_lp_after':am.get('lp'),'monsters_before':bm.get('monsters'),'monsters_after':am.get('monsters')})
            ctx.memory['plan']=note
            ctx.memory['last_progress']=after


def build(ctx): return Harness(ctx)
