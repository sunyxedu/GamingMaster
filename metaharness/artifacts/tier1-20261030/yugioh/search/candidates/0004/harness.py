"""Position-aware actions and checked public-board combat, derived from 0003."""
from __future__ import annotations
import itertools
import json
import re
import time
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def obj(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return obj(json.loads(x))
        except Exception: pass
    return {}


def dump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), default=str)


def number(x, default=0):
    try: return int(x)
    except (ValueError, TypeError): return default


FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
QUICK = {'Book of Moon', 'Rush Recklessly', 'Mystical Space Typhoon'}
TRAPS = {'Trap Hole', 'Mirror Force', 'Compulsory Evacuation Device', 'Call of the Haunted'}


class Board:
    def __init__(self, observation):
        self.o = obj(observation)
        self.p = obj(self.o.get('prompt'))
        self.seat = self.o.get('agent_seat', 0)
        ps = self.o.get('players') or []
        self.me = next((x for x in ps if x.get('player') == self.seat), {})
        self.foe = next((x for x in ps if x.get('player') != self.seat), {})
        self.mine = self.me.get('monsters') or []
        self.enemy = self.foe.get('monsters') or []
        self.catalog = self.o.get('card_catalog') or {}
        self.own = self.o.get('turn_player') == self.seat
        self.main = self.p.get('kind') == 'main_command'
        self.choices = self.p.get('choices') or []
        self.chain = self.o.get('chain') or []
        self.events = self.o.get('recent_events') or []
        self.battle = self.o.get('battle') or self.o.get('current_battle') or {}
        self.strongest = max([number(x.get('attack')) for x in self.mine if 'face-up' in x.get('position','')] + [0])
        self.threat = max([number(x.get('attack')) for x in self.enemy if 'attack' in x.get('position','')] + [0])

    def card(self, ref):
        ref = obj(ref)
        out = dict(self.catalog.get(str(ref.get('code')), {}))
        for pl in (self.me, self.foe):
            for zone in ('hand','monsters','spells_traps','graveyard','banished'):
                for c in pl.get(zone) or []:
                    if all(ref.get(k) == c.get(k) for k in ('player','zone','slot')) and ref.get('zone'):
                        out.update(c)
        out.update(ref)
        return out

    def value(self, card):
        c = self.card(card)
        atk = number(c.get('attack'))
        if c.get('name') in FLIPS: return 1400
        return 650 + atk * .65

    def battle_pair(self):
        # Public battle structures vary: events provide a second source.
        source = self.battle
        if not source:
            for e in reversed(self.events):
                if e.get('event') in ('damage','new_turn','phase'):
                    break
                if e.get('event')=='hint' and any(t in str(e.get('text','')).lower() for t in ('after damage','end of damage','end the battle')):
                    break
                if e.get('event') in ('attack', 'attack_declared', 'battle'):
                    source = e; break
        source = obj(source)
        a = source.get('attacker') or source.get('attacking_card') or {}
        t = source.get('target') or source.get('defender') or source.get('attack_target') or {}
        return self.card(a), self.card(t)

    def effect(self):
        hint = str(self.p.get('hint',''))
        for code,c in self.catalog.items():
            if code in hint or c.get('name','~') in hint: return c.get('name','')
        if self.chain:
            c = obj(self.chain[-1])
            return self.card(c.get('card') or c.get('source') or c).get('name','')
        return ''


def constraints(p):
    c = dict(p); c.update(obj(p.get('constraints')))
    lo = number(c.get('min', c.get('minimum', c.get('min_count',0))))
    hi = number(c.get('max', c.get('maximum', c.get('max_count',len(p.get('candidates') or [])))))
    return lo,hi,c


def ids_of(p):
    return [number(obj(c).get('index'), i) for i,c in enumerate(p.get('candidates') or [])]


def valid(action, p):
    if action in [c.get('action') for c in p.get('choices') or []]: return action
    m = re.fullmatch(r'(select|order)\s+(\S+)\s+(-|\d+(?:,\d+)*)', str(action))
    if not m or m[2] != str(p.get('token')): return None
    ids = [] if m[3]=='-' else list(map(int,m[3].split(',')))
    is_order=('order' in str(p.get('kind','')).lower() or p.get('mode')=='order'
              or str(p.get('grammar','')).startswith('order '))
    if (m[1]=='order')!=is_order: return None
    avail = ids_of(p)
    if len(ids) != len(set(ids)) or any(i not in avail for i in ids): return None
    if m[1]=='order': return action if sorted(ids)==sorted(avail) else None
    lo,hi,c = constraints(p)
    if len(ids)>hi: return None
    tribute = ('tribute' in str(p.get('kind','')).lower() or 'tribute' in str(c.get('mode','')).lower()
               or 'weight' in str(p.get('constraint','')).lower())
    if tribute:
        ws=p.get('weights') or []
        weights = {i:number(ws[n] if n<len(ws) else obj(x).get('weight',obj(x).get('tribute_value',1)),1)
                   for n,(i,x) in enumerate(zip(avail,p.get('candidates') or []))}
        if sum(weights[i] for i in ids)<lo: return None
    elif len(ids)<lo: return None
    return action


def parse_json(text):
    s=str(text)
    try: return obj(json.loads(s.strip().strip('`').removeprefix('json').strip()))
    except Exception: pass
    start=s.find('{')
    if start>=0:
        try: return obj(json.JSONDecoder().raw_decode(s[start:])[0])
        except Exception: pass
    return {}


def exact_action(text,p):
    for m in re.finditer(r'\b(?:choose\s+\S+\s+\d+|(?:select|order)\s+\S+\s+(?:-|\d+(?:,\d+)*))',str(text)):
        a=valid(m[0],p)
        if a: return a
    return None



def loc(card):
    """A current-zone reference, never a permanent identity."""
    return (card.get('player'), card.get('zone'), card.get('slot'), card.get('code'))


def public_combat(b, attackers):
    """Enumerate the tiny visible battle assignment; do not simulate hidden cards.

    Known ATK/DEF arithmetic is exact for this pool absent interaction. A hidden
    defender is an uncertain leaf, not an invented monster which gets removed.
    Return only the first attacker/target; replan after each native response.
    """
    enemies=b.enemy
    atk=[number(c.get('attack')) for c in attackers]
    own=[b.value(c) for c in attackers]
    ev=[b.value(c) for c in enemies]
    start_lp=number(b.me.get('lp'),8000); foe_lp=number(b.foe.get('lp'),8000)
    cache={}; visits=[0]
    def walk(ready,alive,foes,damage,hurt):
        key=(ready,alive,foes,damage,hurt)
        if key in cache: return cache[key]
        visits[0]+=1
        lost=sum(own[i] for i in range(len(atk)) if not alive>>i&1)
        removed=sum(ev[j] for j in range(len(enemies)) if not foes>>j&1)
        value=removed*1.08-lost+damage*.32-hurt*.42
        if damage>=foe_lp: return (1000000+value,None,None,damage)
        if hurt>=start_lp: return (-1000000+value,None,None,damage)
        # The exposed battlefield is a lower bound on next-turn danger.
        remaining_threats=[number(e.get('attack')) for j,e in enumerate(enemies) if foes>>j&1 and 'attack' in e.get('position','')]
        own_max=max([atk[i] for i in range(len(atk)) if alive>>i&1]+[number(c.get('defense')) for c in b.mine if 'defense' in c.get('position','')]+[0])
        value-=sum(max(0,x-own_max) for x in remaining_threats)*.20
        best=(value,None,None,damage)
        if visits[0]>12000:
            return best
        for i,a in enumerate(atk):
            if not ready>>i&1 or not alive>>i&1: continue
            rem=ready&~(1<<i)
            if not foes:
                child=walk(rem,alive,foes,damage+a,hurt)
                if child[0]>best[0]: best=(child[0],i,None,child[3])
                continue
            for j,e in enumerate(enemies):
                if not foes>>j&1: continue
                if 'face-down' in e.get('position',''):
                    # Probe a hidden defender only at an explicit uncertainty
                    # boundary. 2000 DEF is a pool prior, NOT a known identity.
                    probe=100+min(a,2200)*.06-max(0,2000-a)*.6
                    if a<1400: probe-=450
                    score=value+probe
                    if score>best[0]: best=(score,i,j,damage)
                    continue
                defending='defense' in e.get('position','')
                stat=number(e.get('defense') if defending else e.get('attack'))
                delta=a-stat; new_alive=alive; new_foes=foes; dealt=0; taken=0
                if delta>0:
                    new_foes&=~(1<<j)
                    if not defending: dealt=delta
                elif delta==0 and not defending:
                    new_alive&=~(1<<i); new_foes&=~(1<<j)
                elif delta<0:
                    taken=-delta
                    if not defending: new_alive&=~(1<<i)
                # A harmless bounce off known defense has no purpose.
                if defending and delta<=0: continue
                child=walk(rem,new_alive,new_foes,damage+dealt,hurt+taken)
                if child[0]>best[0]: best=(child[0],i,j,child[3])
        cache[key]=best
        return best
    value,i,j,damage=walk((1<<len(atk))-1,(1<<len(atk))-1,(1<<len(enemies))-1,0,0)
    return {'value':round(value,2),'attacker':attackers[i] if i is not None else None,
            'target':enemies[j] if j is not None else None,'damage':damage,
            'assumptions':'No new effects; hidden defender is an uncertain stopping leaf. Only battle-command menus establish actual attack availability.',
            'nodes':visits[0]}


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx
        self.plan=[]
        self.policy={}
        self.last_effect=''
        self.calls=0
        self.round_calls={}
        self.reviewed=set()
        self.last_review_turn=-10
        self.failures=[]
        self.bad_actions={}
        self.last_round=None
        self.last_action=None
        self.last_round_state=None
        self.max_calls=9
        self.pending=None
        self.attacker={}
        self.semantic_visits={}
        self.cycle_reviewed=set()
        self.plan_failed=False
        self.turn_calls={}
        self.prompt_started=time.monotonic()
        self.attack_intent=None
        self.expected_position=None
        self.consequence_failed=False

    def subset(self,b):
        p=b.p; candidates=p.get('candidates') or []; token=p.get('token','')
        ids=ids_of(p)
        if 'order' in str(p.get('kind','')) or p.get('mode')=='order' or str(p.get('grammar','')).startswith('order '):
            return 'order %s %s'%(token,','.join(map(str,ids)))
        lo,hi,c=constraints(p)
        kind=str(p.get('kind','')).lower()
        effect=b.effect() or self.last_effect
        if 'attack target' in str(p.get('hint','')).lower() and self.attack_intent:
            for i,raw in zip(ids,candidates):
                card=b.card(obj(raw).get('card',raw))
                if loc(card)==self.attack_intent:
                    action='select %s %s'%(token,i)
                    if valid(action,p): return action
        ranked=[]
        for i,raw in zip(ids,candidates):
            card=b.card(obj(raw).get('card',raw)); own=card.get('player')==b.seat
            value=b.value(card)
            if 'attack target' in str(p.get('hint','')).lower():
                atk=number(self.attacker.get('attack'),b.strongest)
                if 'face-down' in card.get('position',''): score=500
                elif 'defense' in card.get('position',''):
                    score=1100 if atk>number(card.get('defense')) else -1000
                else:
                    delta=atk-number(card.get('attack'))
                    score=1500+delta if delta>0 else (100 if delta==0 else -3000+delta)
            elif 'tribute' in kind or 'tribute' in str(p.get('mode','')).lower() or 'weight' in str(p.get('constraint','')).lower() or 'discard' in str(p.get('hint','')).lower():
                score=-value
                if card.get('name') in FLIPS and 'face-down' in card.get('position',''): score-=700
            elif effect in ('Monster Reborn','Call of the Haunted'):
                score=number(card.get('attack')) + (100 if own else 0)
            elif effect=='Magician of Faith':
                score={'Monster Reborn':3200,'Dark Hole':2800 if len(b.enemy)>len(b.mine) else 1300,'Fissure':2200,'Book of Moon':1700,'Mystical Space Typhoon':1500,'Rush Recklessly':1200}.get(card.get('name'),500)
            elif effect=='Rush Recklessly':
                a,t=b.battle_pair()
                fighter=a if a.get('player')==b.seat else t
                engaged=own and fighter.get('zone')=='monsters' and card.get('slot')==fighter.get('slot')
                score=(10000 if engaged else value) if own else -value-1000
            elif effect=='Book of Moon':
                score=value if not own else (-value if card.get('name') not in FLIPS else 300)
            elif effect=='Penguin Soldier': score=value if not own else (700 if card.get('name')=='Penguin Soldier' else -value)
            else: score=value if not own else -value-1000
            ranked.append((score,i))
        score_by=dict((i,s) for s,i in ranked)
        best=None
        # Pool has tiny selections (<=10 field cards, <=80 GY cards). For
        # one/two targets exhaustive ranking is cheap; bound larger subsets.
        ranked_ids=[i for _,i in sorted(ranked,reverse=True)]
        for n in range(0,min(hi,len(ids))+1):
            if n>6: break
            combos=itertools.combinations(ranked_ids,n)
            for j,chosen in enumerate(combos):
                if j>10000: break
                a='select %s %s'%(token,','.join(map(str,chosen)) if chosen else '-')
                if not valid(a,p): continue
                val=sum(score_by[i] for i in chosen)
                if best is None or val>best[0]: best=(val,a)
        if best: return best[1]
        # Exact count prompts can be larger than six, e.g. hand discard.
        for n in range(max(0,lo),min(hi,len(ids))+1):
            a='select %s %s'%(token,','.join(map(str,ranked_ids[:n])) if n else '-')
            if valid(a,p): return a
        return None

    def activation_score(self,b,c):
        card=b.card(c.get('card')); name=card.get('name','')
        chain=b.chain
        a,t=b.battle_pair()
        attacked=bool(a.get('zone')) and ('Battle' in str(b.o.get('phase')) or 'Damage' in str(b.o.get('phase')))
        own_turn=b.own
        # A chain already containing our removal is not an invitation to spend
        # a second copy at the same threat. Mandatory triggers still resolve.
        if chain and name not in FLIPS and name not in ('Book of Moon','Rush Recklessly','Mystical Space Typhoon'):
            return -15
        if name=='Mirror Force': return 100 if b.enemy else -15
        if name=='Trap Hole': return 90 if b.enemy else -15
        if name in FLIPS: return 95
        if name=='Fissure':
            faceup=[x for x in b.enemy if 'face-up' in x.get('position','')]
            if not faceup: return -20
            weak=min(faceup,key=lambda x:number(x.get('attack')))
            return 78 if number(weak.get('attack'))>=b.strongest or len(b.enemy)>=len(b.mine) else 24
        if name=='Dark Hole':
            gain=sum(b.value(x) for x in b.enemy)-sum(b.value(x) for x in b.mine)
            return 100+gain/100 if gain>550 or (not b.mine and b.enemy) else -30
        if name in ('Monster Reborn','Call of the Haunted'):
            grave=list(b.me.get('graveyard') or [])
            if name=='Monster Reborn': grave+=list(b.foe.get('graveyard') or [])
            strength=max([number(x.get('attack')) for x in grave if number(x.get('type'))&1]+[0])
            if strength<1500 or len(b.mine)>=4: return -15
            if own_turn or not b.mine or strength>b.threat: return 75+strength/150
            return -5
        if name=='Mystical Space Typhoon':
            back=b.foe.get('spells_traps') or []
            # Do not attempt to negate normal spells/traps by destroying source.
            if not back: return -30
            if chain: return 40 if any(x.get('name')=='Call of the Haunted' for x in back) else -10
            return 65 if own_turn and (b.mine or b.me.get('hand')) else -5
        if name=='Compulsory Evacuation Device':
            if not b.enemy: return -30
            if attacked or b.threat>b.strongest or any(number(x.get('level'))>=5 for x in b.enemy): return 70
            return 25 if own_turn and b.mine else -5
        if name=='Book of Moon':
            if attacked and a.get('player')!=b.seat: return 80
            # Pre-combat book creates an attackable defender (no needless books).
            profitable=any('attack' in x.get('position','') and number(x.get('attack'))>=b.strongest>number(x.get('defense')) for x in b.enemy)
            return 68 if own_turn and b.strongest and profitable else -15
        if name=='Rush Recklessly':
            if attacked:
                mine=a if a.get('player')==b.seat else t
                other=t if a.get('player')==b.seat else a
                if mine.get('player')==b.seat and other:
                    attack=number(mine.get('attack')); opposing=number(other.get('attack'))
                    if 'defense' in other.get('position',''): opposing=number(other.get('defense'))
                    if attack<=opposing<attack+700: return 100
            return -20
        return 5

    def rank_choice(self,b,c):
        label=str(c.get('label','')).lower(); card=b.card(c.get('card')); name=card.get('name','')
        atk=number(card.get('attack')); defense=number(card.get('defense')); level=number(card.get('level'))
        if 'shuffle' in label: return -10000
        if 'decline' in label or label in ('no','cancel','finish'): return 0
        if 'activate' in label or 'activation' in label: return self.activation_score(b,c)
        if 'normal summon' in label:
            if name in FLIPS: return -20
            if level>4:
                cost=1 if level<=6 else 2
                bodies=sorted([b.value(x) for x in b.mine])[:cost]
                if len(bodies)<cost: return -20
                usable=sum(number(x.get('attack')) for x in b.mine if 'attack' in x.get('position',''))
                if cost==2 and usable>=atk and b.threat<usable/2: return -10
                return 65+(atk-b.threat)/100-(sum(bodies)/100)
            return 70+atk/100 if atk>=b.threat or atk>=1700 else 10
        if 'set monster' in label:
            if level>4: return -10
            if name=='Magician of Faith':
                useful=any(number(x.get('type'))&2 for x in b.me.get('graveyard') or [])
                return 78 if useful else 18
            if name in FLIPS: return 105 if b.enemy and b.threat>=b.strongest else 60
            return 63+defense/200 if defense>atk and b.threat>=atk else 12
        if 'set spell' in label or 'set spell/trap' in label:
            if name in TRAPS or name in QUICK:
                if name in QUICK and any('battle phase' in str(x.get('label','')).lower() for x in b.choices):
                    return -4
                return 40 if len(b.me.get('spells_traps') or [])<4 else 10
            return -10
        if 'flip summon' in label or 'change position' in label or 'position change' in label:
            # One native label covers three DIFFERENT state transitions.
            # Inspect the actual current position before assigning any value.
            position = str(card.get('position', ''))
            can_battle = any('battle phase' in str(x.get('label','')).lower() for x in b.choices)
            if 'face-down' in position:
                if name=='Man-Eater Bug': return 95 if b.enemy else -25
                if name=='Penguin Soldier': return 93 if b.enemy else -20
                if name=='Magician of Faith': return 92 if any(number(x.get('type'))&2 for x in b.me.get('graveyard') or []) else -20
                return 58 if can_battle and (not b.enemy or atk>=b.threat) else -20
            if 'attack' in position:
                # Keep useful attackers upright. A weak defender can switch
                # after combat, or when it has no productive legal attack.
                productive = not b.enemy or any('face-down' in x.get('position','') or atk >= number(x.get('attack') if 'attack' in x.get('position','') else x.get('defense')) for x in b.enemy)
                if can_battle and productive: return -25
                if defense>atk and (atk<b.threat or atk<1500): return 55
                if not can_battle and b.threat>atk and number(b.me.get('lp')) <= b.threat-atk: return 55
                return -25
            if 'defense' in position:
                if not can_battle: return -25
                if not b.enemy: return 58
                profitable = any('face-up' in x.get('position','') and atk>number(x.get('attack') if 'attack' in x.get('position','') else x.get('defense')) for x in b.enemy)
                return 58 if profitable or atk>=b.threat and atk>=1500 else -20
            return -25
        if label.startswith('attack') or 'direct attack' in label:
            # Command chooses attacker; target is supplied by a following menu.
            opponents=b.enemy
            if not opponents: return 65+atk/100
            values=[]
            for x in opponents:
                if 'face-down' in x.get('position',''): values.append(25+atk/300)
                elif 'attack' in x.get('position',''):
                    delta=atk-number(x.get('attack'))
                    values.append(70+delta/100 if delta>0 else (15 if delta==0 else -50))
                else: values.append(65 if atk>number(x.get('defense')) else -30)
            return max(values+[-30])
        if 'battle phase' in label: return 20 if any('attack' in x.get('position','') for x in b.mine) else -5
        if 'main phase 2' in label or 'main 2' in label: return 0
        if 'end turn' in label or 'end phase' in label: return -1
        if 'face-up attack' in label: return 10
        if 'face-up defense' in label: return 5
        return 0

    def local(self,b):
        p=b.p; choices=b.choices; kind=str(p.get('kind',''))
        # Subset prompts also contain an optional cancel choice. It is not a
        # singleton mandatory command: select from candidates before choices.
        if p.get('mode') in ('subset','order') or p.get('candidates'):
            selected=self.subset(b)
            if selected: return selected
        if not choices: return self.subset(b)
        if len(choices)==1: return choices[0].get('action')
        if kind=='battle_command':
            attacks=[c for c in choices if str(c.get('label','')).lower()=='attack']
            if attacks:
                result=public_combat(b,[b.card(c.get('card')) for c in attacks])
                if result['attacker']:
                    chosen=next(c for c in attacks if loc(b.card(c.get('card')))==loc(result['attacker']))
                    # A profitable immediately available effect may precede battle.
                    effects=[c for c in choices if 'activat' in str(c.get('label','')).lower() and self.activation_score(b,c)>50]
                    if effects: return max(effects,key=lambda c:self.activation_score(b,c)).get('action')
                    self.attack_intent=loc(result['target']) if result['target'] else None
                    return chosen.get('action')
                exit_choice=next((c for c in choices if 'main phase 2' in str(c.get('label','')).lower() or 'end turn' in str(c.get('label','')).lower()),None)
                if exit_choice: return exit_choice.get('action')
        if 'zone' in kind or 'place' in kind: return choices[0].get('action')
        if 'position' in kind:
            effect=self.last_effect
            return max(choices,key=lambda c:self.rank_choice(b,c)).get('action')
        if 'target' in kind or 'card' in kind and kind not in ('main_command','battle_command'):
            # Choice-style targeting, if used by an adaptor revision.
            effect=b.effect() or self.last_effect
            def value(c):
                card=b.card(c.get('card',c)); v=b.value(card)
                if effect in ('Monster Reborn','Call of the Haunted','Rush Recklessly','Magician of Faith'): return v
                return v if card.get('player')!=b.seat else -v
            return max(choices,key=value).get('action')
        if kind in ('yes_no','effect_yes_no','yesno'):
            effect=b.effect() or self.last_effect
            yes=effect in FLIPS or 'replay' in str(p.get('hint','')).lower()
            return next((c.get('action') for c in choices if str(c.get('label','')).lower()==('yes' if yes else 'no')),choices[-1].get('action'))
        ranked=sorted(choices,key=lambda c:self.rank_choice(b,c),reverse=True)
        blocked=self.bad_actions.get(p.get('token'),set())
        return next((c.get('action') for c in ranked if c.get('action') not in blocked),ranked[0].get('action'))

    def compile(self,b):
        if not b.main: return None
        while self.plan:
            intent=self.plan[0]
            if not isinstance(intent,dict): self.plan.pop(0); continue
            verb=str(intent.get('label',intent.get('op',''))).lower()
            name=str(intent.get('card',intent.get('name','')))
            aliases={'summon':'normal summon','set':'set monster','activate':'activate','battle':'enter battle phase','end':'end turn','flip':'flip summon','backrow':'set spell/trap'}
            verb=aliases.get(verb,verb)
            matches=[c for c in b.choices if verb and verb in str(c.get('label','')).lower() and (not name or name==obj(c.get('card')).get('name'))]
            if 'flip summon' in verb or 'position' in verb:
                # Future prose must express the SAME transition as the current
                # card state, not merely share the menu's ambiguous label.
                destination=str(intent.get('to','')).lower()
                if verb=='flip summon':
                    matches=[c for c in matches if 'face-down' in b.card(c.get('card')).get('position','')]
                elif destination in ('face-up attack','face-up defense'):
                    matches=[c for c in matches if ('face-up defense' if 'attack' in b.card(c.get('card')).get('position','') else 'face-up attack')==destination]
                else:
                    matches=[]
            if matches:
                self.plan.pop(0)
                return matches[0].get('action')
            # An intention can become impossible after an opponent response.
            self.failures.append('Intention unavailable: '+dump(intent))
            self.plan_failed=True
            self.plan=[]
        return None

    def compact(self,b):
        out={k:v for k,v in b.o.items() if k not in ('card_catalog','recent_events')}
        out['recent_events']=b.events[-12:]
        out['effect_texts']={v['name']:v.get('text','') for v in b.catalog.values() if number(v.get('type'))!=17}
        out['suggested_action']=self.local(b)
        out['pending_intentions']=self.plan
        out['position_semantics']='The menu label change position / flip summon flips ONLY face-down monsters; a face-up attacker instead changes to defense.'
        out['combat_without_new_interaction']=public_combat(b,[c for c in b.mine if 'attack' in c.get('position','')])
        out['previous_position_check']=self.expected_position
        return out

    def ask(self,game,b,reason):
        ctx=self.ctx; remaining=ctx.budget.remaining()
        # Backend queueing may not be included in reported model latency.
        # Never block a native pending decision behind an in-flight request.
        if self.pending is not None:
            if not self.pending.done(): return None
            try: self.pending.result()
            except Exception: pass
            self.pending=None
        if self.calls>=self.max_calls or remaining.get('system2_calls',0)<=1 or remaining.get('wall_s',0)<260: return None
        # Retries share the native 300-second deadline for this prompt.
        wait_allowance=min(210.0,240.0-(time.monotonic()-self.prompt_started))
        if wait_allowance<115: return None
        self.calls+=1
        r=b.o.get('round',1); self.round_calls[r]=self.round_calls.get(r,0)+1
        key=(r,b.o.get('turn')); self.turn_calls[key]=self.turn_calls.get(key,0)+1
        self.plan_failed=False
        self.reviewed.add((r,b.o.get('turn'))); self.last_review_turn=number(b.o.get('turn'))
        prompt=(
            'You control a three-duel Yu-Gi-Oh fixed-pool game. Only visible information is available. '
            'A native engine supplies legal actions. We need a SHORT plan for this turn; code handles '
            'zones, target selections, conservative reaction timing, battle arithmetic and fresh token binding. '
            'IMPORTANT: change position / flip summon is a shared label: an upright attacker changes to DEFENSE, not a Flip Summon. Only face-down monsters Flip Summon. Keep strong attackers ready for battle. Prioritize winning and preventing loss, then card advantage and pressure. Preserve Flip monsters '
            'face-down unless their effect is useful. MST does not negate an activated ordinary spell/trap. '
            'Use removal before summoning into Dark Hole; avoid tributing productive attackers just for stats. '
            'RULES: one Normal Summon OR Normal Set per turn, not both. Level 5-6 needs one tribute, '
            '7+ needs two. A Normal Spell (Fissure, Dark Hole, Monster Reborn) cannot be activated '
            'on the opponent turn even if Set; keep these in hand until useful. Set Traps and quick-play '
            'Spells for opponent-turn use. Newly Set quick-play Spells cannot activate that same turn; '
            'leave Rush/Book/MST in hand if you need them this Battle Phase. First player cannot battle turn 1. '
            'Unknown backrow can punish overextension; do not assume a hidden card identity.\n'
            'Return JSON only: {"action":"one exact current choose/select/order action",'
            '"plan":[{"label":"exact main command label such as set spell/trap","card":"exact name"}],'
            '"note":"brief tactical reason"}. plan is at most SIX FUTURE main-command intentions for THIS '
            'turn after action. Omit card for phase transitions. Do not invent future tokens. '
            'Code recompiles each intention against the current legal menu; it discards stale plans. '
            'For any future position-change intention include to: face-up attack or face-up defense. Only request flip summon on a face-down monster. For subset prompts obey displayed count/weight constraints. '
            'If no useful future plan, return []. Keep response under 220 words.\n'
            'Review trigger: '+reason+'\nFailure evidence: '+dump(self.failures[-4:])+
            '\nProgress: '+dump(ctx.progress())+'\nState: '+dump(self.compact(b)))
        try:
            started=time.monotonic()
            clock_started=ctx.clock.now()
            self.pending=ctx.spawn(lambda:ctx.think(prompt,session_id=None,timeout_s=235.0))
            while not self.pending.done():
                if ctx.done: return None
                if max(time.monotonic()-started,ctx.clock.now()-clock_started)>wait_allowance:
                    self.failures.append('Planner wait exceeded 210s; live policy resumes')
                    ctx.log('planner_wait_guard',token=b.p.get('token'))
                    return None
                time.sleep(.025)
            result=self.pending.result()
            self.pending=None
            parsed=parse_json(result.text)
            plan=parsed.get('plan',[])
            self.plan=plan[:6] if isinstance(plan,list) else []
            self.policy={'note':str(parsed.get('note',''))[:300]}
            ctx.memory['last_plan']=self.plan
            ctx.log('plan_review',reason=reason,plan=self.plan,note=self.policy['note'])
            action=valid(str(parsed.get('action','')),b.p) or exact_action(result.text,b.p)
            if not action: self.plan=[]
            self.failures=[]
            return action
        except (BudgetExceeded,ModelUnavailable): return None
        except (TimeoutError,RuntimeError) as exc:
            self.failures.append('Model failure '+type(exc).__name__)
            ctx.log('model_recovery',error=type(exc).__name__)
            return None
        except Exception as exc:
            self.failures.append('Model parse failure '+type(exc).__name__)
            return None

    def run_episode(self,game):
        ctx=self.ctx; obs=game.observe(); previous=None; stall=0; steps=0
        while not ctx.done and not getattr(game,'done',False):
            b=Board(obs); r=b.o.get('round'); turn=number(b.o.get('turn')); token=b.p.get('token')
            if token!=previous:
                self.prompt_started=time.monotonic(); previous=token
            progress=ctx.progress()
            if r!=self.last_round:
                self.plan=[]; self.last_effect=''; self.reviewed=set(); self.last_review_turn=-10
                self.failures=[]; self.last_round=r
                self.semantic_visits={}; self.cycle_reviewed=set()
                self.plan_failed=False
            shape=dump([r,turn,b.o.get('phase'),b.p.get('kind'),b.p.get('hint'),
                        b.o.get('players'),b.chain,b.events[-4:],
                        [{k:v for k,v in c.items() if k!='action'} for c in b.choices],
                        b.p.get('candidates')])
            self.semantic_visits[shape]=self.semantic_visits.get(shape,0)+1
            repeated=self.semantic_visits[shape]>=3
            error=b.o.get('last_action_error')
            if error or progress.get('since_obs_change',0)>1 or stall:
                self.failures.append('No advancement/error: '+dump(error or progress))
                self.plan=[]
            reason=None
            if self.consequence_failed:
                reason='Observed action consequence differed from planned position; repair'
                self.consequence_failed=False
            elif stall and stall<=2: reason='Rejected or stalled previous action'
            elif repeated and shape not in self.cycle_reviewed:
                self.plan=[]
                self.cycle_reviewed.add(shape)
                self.failures.append('Same decision and public board repeated despite new prompt tokens')
                reason='Semantic action loop; reconsider and force actual progress'
            elif self.calls==0 and b.main: reason='Review opening main-phase plan and position consequences'
            elif b.main and self.plan_failed and self.turn_calls.get((r,turn),0)<2:
                reason='Previous semantic intention became unavailable; repair the plan'
            elif b.main and (r,turn) not in self.reviewed and self.round_calls.get(r,0)<2:
                if not self.plan: reason='Plan the current own turn'
            action=None
            if reason: action=self.ask(game,b,reason)
            if ctx.done: break
            if not action: action=self.compile(b)
            if not action and self.plan_failed and b.main and self.turn_calls.get((r,turn),0)<2:
                action=self.ask(game,b,'Semantic intention no longer legal; repair the plan')
            if not action: action=self.local(b)
            if repeated and self.semantic_visits[shape]>3 and b.p.get('kind') in ('main_command','battle_command'):
                action=next((c.get('action') for c in b.choices if 'main phase 2' in str(c.get('label','')).lower()
                             or 'end turn' in str(c.get('label','')).lower()),action)
            if not action:
                action=self.ask(game,b,'Unknown prompt; supply legal mandatory answer')
                if not action:
                    # No stale/noop loop: all known prompt types use validators.
                    action=next((c.get('action') for c in b.choices if c.get('action')),None)
                    if not action: break
            chosen=next((c for c in b.choices if c.get('action')==action),{})
            label=str(chosen.get('label','')).lower()
            self.expected_position=None
            if 'change position' in label or 'flip summon' in label:
                card=b.card(chosen.get('card'))
                position=str(card.get('position',''))
                destination='face-up defense' if 'attack' in position else 'face-up attack'
                self.expected_position={'ref':loc(card),'from':position,'to':destination}
            if label=='attack': self.attacker=b.card(chosen.get('card'))
            if label=='yes' and chosen.get('card'): self.last_effect=b.card(chosen.get('card')).get('name','')
            if 'activat' in label: self.last_effect=b.card(chosen.get('card')).get('name','')
            before_progress=ctx.progress()
            try: new=ctx.act(action)
            except (BudgetExceeded,ModelUnavailable): break
            except Exception as exc:
                self.failures.append(type(exc).__name__+': '+str(exc)[:160])
                self.bad_actions.setdefault(token,set()).add(action)
                self.plan=[]; stall+=1
                try: obs=ctx.observe()
                except Exception: break
                if stall>8: break
                continue
            if new is None: break
            after=ctx.progress(); nb=Board(new)
            # Verify every local/model action: reject unchanged prompt and newly
            # counted illegal actions. Never use cumulative illegals as a latch.
            bad=(nb.p.get('token')==token and not nb.o.get('done')) or after.get('illegal_actions',0)>before_progress.get('illegal_actions',0) or bool(nb.o.get('last_action_error'))
            stall=stall+1 if bad else 0
            if bad:
                self.bad_actions.setdefault(token,set()).add(action); self.plan=[]
                self.failures.append('Action failed '+str(action)+' '+dump(after))
            if self.expected_position and nb.o.get('round')==r:
                exp=self.expected_position
                actual=next((c for c in nb.mine if loc(c)==tuple(exp['ref'])),None)
                if actual is not None and actual.get('position')!=exp['to']:
                    self.plan=[]; self.plan_failed=True; self.consequence_failed=True
                    self.failures.append('Position consequence mismatch: '+dump(exp)+' actual='+str(actual.get('position')))
                    ctx.log('consequence_mismatch',expected=exp,actual=actual.get('position'))
                elif actual is not None:
                    ctx.log('position_verified',card=actual.get('name'),before=exp['from'],after=exp['to'])
                else:
                    # A response may destroy/bounce the body. Never pretend it
                    # achieved the intended combat role merely because token moved.
                    self.plan=[]; self.plan_failed=True; self.consequence_failed=True
                    self.failures.append('Position-changing body left field; replan against response')
            # A turn or round boundary invalidates every unused intention.
            if nb.o.get('turn')!=b.o.get('turn') or nb.o.get('round')!=r: self.plan=[]
            # Losing a friendly body or a large LP amount indicates interaction
            # that can invalidate the intended line; replan next main command.
            loss=len(b.mine)-len(nb.mine)
            lp_loss=number(b.me.get('lp'))-number(nb.me.get('lp'))
            if (loss>0 and 'normal summon' not in label) or lp_loss>=1800:
                self.plan=[]
                self.reviewed.discard((r,turn))
                self.failures.append('Unexpected board/resource loss after '+str(action))
            self.last_action=action; steps+=1; obs=new
            if stall>8: break


def build(ctx): return Harness(ctx)
