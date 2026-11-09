"""Prompt-exact Yu-Gi-Oh controller; code handles mechanics, Astra scarce dilemmas."""
from __future__ import annotations
import itertools
import json
import re
from metaharness.core.contract import BudgetExceeded, ModelUnavailable

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
QUICK = {'Mystical Space Typhoon', 'Book of Moon', 'Rush Recklessly'}


def number(x, default=0):
    try: return int(x)
    except (ValueError, TypeError): return default


def unpack(obs):
    if isinstance(obs, str):
        try: return json.loads(obs)
        except (ValueError, TypeError): return {}
    return obs if isinstance(obs, dict) else {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = {}
        self.me = {}; self.opp = {}; self.catalog = {}
        self.source = ''; self.attacker = None; self.selected = None
        self.round = None; self.calls = 0; self.advised = set()
        self.bad = {}; self.steps = 0; self.pending = None

    def card(self, c):
        c = c or {}
        if 'card' in c: c = c['card']
        base = dict(self.catalog.get(str(c.get('code')), {}))
        base.update(c)
        for p in self.obs.get('players', []):
            for z in ('hand', 'monsters', 'spells_traps', 'graveyard'):
                for real in p.get(z, []):
                    if all(c.get(k) == real.get(k) for k in ('player', 'zone', 'slot')):
                        base.update(real)
        return base

    def value(self, c):
        c = self.card(c)
        if c.get('name') == 'unknown card': return 1900
        a, d = number(c.get('attack')), number(c.get('defense'))
        return 600 + max(a, d * .65) + (700 if c.get('name') in FLIPS else 0)

    def battle(self, attacker, defender):
        a = number(self.card(attacker).get('attack'))
        d = self.card(defender)
        if d.get('name') == 'unknown card':
            return 750 if a >= 2000 else 250 if a >= 1700 else -500
        if 'defense' in d.get('position', ''):
            defense = number(d.get('defense'))
            return self.value(d) if a > defense else -max(0, defense-a)*.65-100
        defense = number(d.get('attack'))
        if a > defense: return self.value(d)+(a-defense)*.7
        if a == defense: return self.value(d)-self.value(attacker)+100
        return -self.value(attacker)-(defense-a)

    def update(self, obs):
        self.obs = unpack(obs)
        self.catalog = self.obs.get('card_catalog', {})
        seat = self.obs.get('agent_seat', 0)
        self.me = next((p for p in self.obs.get('players', []) if p.get('player') == seat), {})
        self.opp = next((p for p in self.obs.get('players', []) if p.get('player') != seat), {})
        if self.round != self.obs.get('round'):
            self.round = self.obs.get('round'); self.calls = 0; self.advised = set()
            self.source = ''; self.attacker = None; self.selected = None
        self.own = self.me.get('monsters', [])
        self.enemy = self.opp.get('monsters', [])
        self.ownturn = self.obs.get('turn_player') == seat

    def effect_score(self, name):
        own, enemy = self.own, self.enemy
        face = [m for m in enemy if 'face-up' in m.get('position', '')]
        strongest = max([number(m.get('attack')) for m in own] or [0])
        threat = max([number(m.get('attack'), 1900) for m in enemy] or [0])
        chain = self.obs.get('chain', [])
        if name == 'Mirror Force': return 1800 if enemy else -100
        if name == 'Trap Hole': return 1400 if enemy else -100
        if name in FLIPS: return 1600
        if name == 'Fissure':
            return 900 + max(0, threat-strongest) if face else -1000
        if name == 'Dark Hole':
            delta = sum(self.value(c) for c in enemy)-sum(self.value(c) for c in own)
            return 1800+delta/5 if enemy and delta > 600 else -1000
        if name in ('Monster Reborn', 'Call of the Haunted'):
            gy = self.me.get('graveyard', []) + (self.opp.get('graveyard', []) if name == 'Monster Reborn' else [])
            targets = [c for c in gy if number(c.get('type')) & 1]
            return 1100+max([self.value(c)/10 for c in targets] or [-2000]) if len(own)<5 else -1000
        if name == 'Mystical Space Typhoon':
            return 1000 if self.opp.get('spells_traps') and not chain else -600
        if name == 'Compulsory Evacuation Device':
            if enemy and (self.ownturn or 'Battle' in self.obs.get('phase','') or threat>=2500): return 950
            return -300
        if name == 'Book of Moon':
            if not face: return -500
            if not self.ownturn and 'Battle' in self.obs.get('phase', ''): return 700
            if self.ownturn and any(number(c.get('attack')) >= strongest > number(c.get('defense')) for c in face): return 700
            return -400
        if name == 'Rush Recklessly':
            if 'Battle' not in self.obs.get('phase', '') and 'Damage' not in self.obs.get('phase', ''): return -500
            if own and enemy and any(0 <= number(e.get('attack'))-number(c.get('attack')) <700 for c in own for e in face): return 650
            if own and not enemy and sum(number(c.get('attack')) for c in own)+700 >= self.opp.get('lp',8000): return 800
            return -300
        return -100

    def score(self, choice, prompt):
        label = choice.get('label', '').lower()
        c = self.card(choice.get('card', {})); name = c.get('name', '')
        a = number(c.get('attack')); d=number(c.get('defense')); level=number(c.get('level'))
        if 'shuffle' in label: return -9999
        if 'decline' in label or 'cancel' in label or 'finish' in label: return 0
        if 'end turn' in label: return 1
        if 'main phase 2' in label or 'end battle' in label: return 2
        if 'enter battle' in label: return 20 if any('attack' in m.get('position','') for m in self.own) else -10
        if label in ('activate','chain activation') or 'activate effect' in label: return self.effect_score(name)
        if 'normal summon' in label:
            if name in FLIPS: return 30+a/30
            val = 450+a/10
            if level>4:
                n=1 if level<=6 else 2
                bodies=sorted(self.own,key=self.value)[:n]
                if len(bodies)<n: return -1000
                val-=sum(self.value(m) for m in bodies)*.25
                threat=max([number(m.get('attack'),1900) for m in self.enemy] or [0])
                if a>threat>=max([number(m.get('attack')) for m in self.own] or [0]): val+=750
            return val
        if 'set monster' in label:
            if name == 'Man-Eater Bug': return 650 if self.enemy else 390
            if name == 'Penguin Soldier': return 620 if self.enemy else 380
            if name == 'Magician of Faith': return 560 if any(number(x.get('type'))&2 for x in self.me.get('graveyard',[])) else 220
            return 530+d/25 if a<1500 else 100
        if 'set spell' in label:
            if number(c.get('type'))&4 or name in QUICK: return 320-len(self.me.get('spells_traps',[]))*12
            return -300
        if 'flip summon' in label and 'face-down' in c.get('position',''):
            if name in FLIPS: return 850 if name!='Magician of Faith' or any(number(x.get('type'))&2 for x in self.me.get('graveyard',[])) else 100
            return 500 if a>=1700 else -100
        if 'position' in label and 'change' in label:
            if 'attack' in c.get('position',''):
                return 100 if a<1500 and d>a and self.enemy else -100
            return 350 if a>=1700 else -100
        if 'attack' in label and choice.get('card'):
            if not self.enemy: return 2000+a
            best=max(self.battle(c,m) for m in self.enemy)
            return 100+best if best>=0 else -200
        kind = prompt.get('kind','')
        if 'position' in kind:
            a=number((self.selected or {}).get('attack'),1900)
            if 'attack' in label: return 200 if a>=1500 or not self.enemy else 50
            if 'defense' in label: return 100
        if label in ('yes','no') or label.startswith('yes') or label.startswith('no'):
            return (100 if label.startswith('yes') else 0)
        if choice.get('card'): return self.target_score(c, prompt)
        return 0

    def target_score(self, c, prompt):
        c = self.card(c); mine=c.get('player')==self.obs.get('agent_seat')
        kind=prompt.get('kind',''); hint=str(prompt.get('hint','')).lower()
        val=self.value(c)
        if 'tribute' in kind or 'release' in hint or 'discard' in hint: return -val
        if c.get('zone')=='graveyard':
            if self.source=='Magician of Faith': return {'Monster Reborn':4000,'Dark Hole':3000 if len(self.enemy)>len(self.own) else 900,'Fissure':2700,'Mystical Space Typhoon':2000,'Book of Moon':1500,'Rush Recklessly':1300}.get(c.get('name'),1000)
            return number(c.get('attack'))+val*.1
        if self.attacker and ('attack' in kind or 'attack' in hint): return self.battle(self.attacker,c)
        if self.source=='Rush Recklessly': return val if mine else -val-10000
        if self.source in ('Mystical Space Typhoon','Fissure','Man-Eater Bug','Penguin Soldier','Compulsory Evacuation Device','Book of Moon'):
            return val if not mine else -val-10000
        return val if not mine else -val

    def subset_valid(self, ids, p):
        candidates = p.get('candidates',[])
        available={number(c.get('index'),i):c for i,c in enumerate(candidates)}
        if len(set(ids))!=len(ids) or any(i not in available for i in ids): return False
        lo=number(p.get('min',p.get('minimum',1))); hi=number(p.get('max',p.get('maximum',len(candidates))))
        cons=p.get('constraint',p.get('constraints',{}))
        if isinstance(cons,dict):
            lo=number(cons.get('min',cons.get('minimum',lo))); hi=number(cons.get('max',cons.get('maximum',hi)))
        if 'tribute' in p.get('kind',''):
            return sum(number(available[i].get('weight', (p.get('weights') or [1]*len(candidates))[next(j for j,c in enumerate(candidates) if number(c.get('index'),j)==i)]),1) for i in ids)>=lo and len(ids)<=hi
        return lo<=len(ids)<=hi

    def subset(self,p):
        candidates=p.get('candidates',[])
        indexed=[(number(c.get('index'),i),self.target_score(c,p)) for i,c in enumerate(candidates)]
        indexed.sort(key=lambda x:x[1],reverse=True)
        opts=[]
        if len(indexed)<=16:
            for k in range(len(indexed)+1):
                for sub in itertools.combinations(indexed,k):
                    ids=[x[0] for x in sub]
                    if self.subset_valid(ids,p): opts.append((sum(x[1] for x in sub)-k*50,ids))
        else:
            for k in range(len(indexed)+1):
                ids=[x[0] for x in indexed[:k]]
                if self.subset_valid(ids,p): opts.append((sum(x[1] for x in indexed[:k])-k*50,ids))
        ids=max(opts,key=lambda x:x[0])[1] if opts else [x[0] for x in indexed[:1]]
        return 'select '+p.get('token','')+' '+(','.join(map(str,ids)) if ids else '-')

    def fallback(self,p):
        if p.get('candidates') and (p.get('mode')=='subset' or 'tribute' in p.get('kind','')):
            return self.subset(p)
        choices=p.get('choices',[])
        if choices:
            bad=self.bad.get(p.get('token'),set())
            valid=[c for c in choices if c.get('action') not in bad] or choices
            return max(valid,key=lambda c:self.score(c,p))['action']
        if 'order' in p.get('kind','') or 'sort' in p.get('kind',''):
            return 'order '+p.get('token','')+' '+','.join(str(c.get('index',i)) for i,c in enumerate(p.get('candidates',[])))
        if p.get('candidates') is not None: return self.subset(p)
        return 'noop'

    def parse(self,text,p):
        choices={c.get('action') for c in p.get('choices',[])}
        for m in reversed(list(re.finditer(r'\b(?:choose|select|order)\s+\S+\s+(?:-|\d+(?:\s*,\s*\d+)*)',str(text)))):
            action=re.sub(r'\s*,\s*',',',m.group(0)).strip()
            if action in choices:return action
            bits=action.split()
            if len(bits)!=3 or bits[1]!=p.get('token'):continue
            ids=[] if bits[2]=='-' else [int(i) for i in bits[2].split(',')]
            if bits[0]=='select' and p.get('mode')=='subset' and self.subset_valid(ids,p):return action
            if bits[0]=='order' and ('order' in p.get('kind','') or 'sort' in p.get('kind','') or p.get('mode')=='order') and sorted(ids)==sorted(number(c.get('index'),i) for i,c in enumerate(p.get('candidates',[]))):return action
        return None

    def advise(self,p,fallback):
        key=(self.round,self.obs.get('turn'))
        # At most four strategic consultations per duel, only ambiguous main phases.
        if p.get('kind')!='main_command' or key in self.advised or self.calls>=4:return fallback
        if not self.enemy or len(p.get('choices',[]))<4:return fallback
        if self.pending is not None:
            if not self.pending.done(): return fallback
            try: self.pending.result()
            except Exception: pass
            self.pending = None
        budget=self.ctx.budget.remaining()
        if budget.get('wall_s',0)<180 or budget.get('system2_calls',0)<=1:return fallback
        self.advised.add(key);self.calls+=1
        state={k:self.obs.get(k) for k in ('round','your_deck','agent_seat','turn_player','turn','phase','players','chain')}
        state['recent_events']=self.obs.get('recent_events',[])[-8:]
        known={c.get('name') for pl in self.obs.get('players',[]) for z in ('hand','monsters','spells_traps','graveyard') for c in pl.get(z,[])}
        texts={c.get('name'):c.get('text') for c in self.catalog.values() if c.get('name') in known and number(c.get('type'))!=17}
        prompt=('Play this beginner Yu-Gi-Oh duel against a random-legal opponent. Choose the best CURRENT legal action. '
                'Prioritize native victory, survival, card advantage, then damage. Preserve productive bodies; do not tribute two good attackers just for 3000 ATK. '
                'MST destroys but does not negate. Do not activate cards uselessly. Unknown identities are private. '
                'You need only pick this tactical decision; code handles routine prompts. Return exactly one full action from CURRENT PROMPT.\n'
                'CURRENT PROMPT (unabridged): '+json.dumps(p,separators=(',',':'))+'\n'
                'STATE: '+json.dumps(state,separators=(',',':'))+'\nCARD TEXT: '+json.dumps(texts)+'\n'
                'Scripted suggestion: '+fallback+'\nReturn only the complete action, including token and index.')
        try:
            self.pending=self.ctx.spawn(lambda: self.ctx.think(prompt,session_id=None,timeout_s=min(155,max(10,budget['wall_s']-60))))
            result=self.pending.result(timeout=110)
            self.pending=None
            action=self.parse(result.text,p)
            return action or fallback
        except Exception: return fallback

    def remember(self,action,p):
        if action.startswith('select '):
            tail=action.split()[-1]
            if tail!='-':
                ix=number(tail.split(',')[0],-1)
                cand=next((c for i,c in enumerate(p.get('candidates',[])) if number(c.get('index'),i)==ix),None)
                if cand:self.selected=self.card(cand)
        for c in p.get('choices',[]):
            if c.get('action')==action:
                label=c.get('label','').lower()
                if c.get('card'):self.selected=self.card(c['card'])
                if 'activate' in label or 'activation' in label or 'flip summon' in label or (label=='yes' and p.get('kind')=='activate_effect'):
                    self.source=c.get('card',{}).get('name','')
                if 'attack' in label and c.get('card'):self.attacker=self.card(c['card'])

    def run_episode(self,game):
        obs=self.ctx.observe()
        while not self.ctx.done:
            self.update(obs)
            if self.obs.get('done'):break
            p=self.obs.get('prompt') or {}
            fallback=self.fallback(p)
            action=self.advise(p,fallback)
            self.remember(action,p)
            old=p.get('token')
            try:
                obs=self.ctx.act(action)
            except (BudgetExceeded,ModelUnavailable):break
            except Exception:
                self.bad.setdefault(old,set()).add(action)
                obs=self.ctx.observe()
                continue
            if obs is None:obs=self.ctx.observe()
            new=unpack(obs)
            if new.get('last_action_error') and (new.get('prompt') or {}).get('token')==old:
                if not self.bad.get(old):
                    self.ctx.log('invalid_action', action=action, prompt=new.get('prompt'), error=new.get('last_action_error'))
                self.bad.setdefault(old,set()).add(action)
            self.steps+=1



def build(ctx):return Harness(ctx)
