"""Chain-aware effect transactions with observed-consequence review (0008)."""
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


class BaseHarness:
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

    def run_episode(self,game):
        ctx=self.ctx; obs=game.observe(); previous=None; stall=0; steps=0
        while not ctx.done and not getattr(game,'done',False):
            b=Board(obs); r=b.o.get('round'); turn=number(b.o.get('turn')); token=b.p.get('token')
            if token!=previous:
                self.prompt_started=time.monotonic(); previous=token
            progress=ctx.progress()
            if r!=self.last_round:
                self.transaction=None; self.audits=[]; self.exchange_reviewed=set()
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
            reason=self.review_reason(b)
            if stall and stall<=2:
                reason='Rejected or stalled previous action'
            elif repeated and shape not in self.cycle_reviewed:
                self.cycle_reviewed.add(shape)
                self.failures.append('Same public board and semantic decision repeated')
                reason='Semantic action loop; repair with actual progress'
            action=None
            if reason: action=self.ask(game,b,reason)
            if ctx.done: break
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
            self.record_action(b, action)
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
            self.verify_result(b, nb, action)
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




def refkey(c):
    return (c.get('player'), c.get('zone'), c.get('slot'))


def same_card(a, b):
    return refkey(a)==refkey(b) and (not a.get('code') or not b.get('code') or a['code']==b['code'])


def faceup(c):
    return 'face-up' in str(c.get('position',''))


class Projection:
    """Small public-chain interpreter, not an engine or a hidden-state simulator.

    Tracks only pinned pool effects. Engine menus determine activation legality.
    Ordinary spells still resolve if their source is destroyed. Unknown/missing
    targets are reported, never filled with guessed identities.
    """
    def __init__(self,b,extra=None):
        self.b=b
        self.field={refkey(c):dict(c) for p in (b.me,b.foe)
                    for zone in ('monsters','spells_traps') for c in p.get(zone,[])}
        self.destroyed=[]; self.returned=[]; self.unknown=[]
        links=list(b.chain)
        if extra: links.append(extra)
        for link in reversed(links): self.apply(link)

    def get(self,c):
        return self.field.get(refkey(c),{})

    def remove(self,c,bounce=False):
        actual=self.field.pop(refkey(c),None)
        if actual: (self.returned if bounce else self.destroyed).append(actual)

    def apply(self,link):
        source=self.b.card(link.get('card') or link.get('source') or {})
        name=source.get('name',''); owner=source.get('player')
        targets=link.get('targets') or []
        if name in ('Rush Recklessly','Book of Moon','Compulsory Evacuation Device',
                    'Mystical Space Typhoon','Trap Hole','Man-Eater Bug') and not targets:
            self.unknown.append(name+' target not yet public'); return
        if name=='Rush Recklessly':
            for t in targets:
                c=self.get(t)
                if c and faceup(c): c['attack']=number(c.get('attack'))+700
        elif name=='Book of Moon':
            for t in targets:
                c=self.get(t)
                if c and faceup(c):
                    c['position']='face-down defense'
                    # Turning face-down resets an already applied temporary boost.
                    base=self.b.catalog.get(str(c.get('code')),{}).get('attack')
                    if base is not None: c['attack']=base
        elif name in ('Compulsory Evacuation Device','Penguin Soldier'):
            for t in targets: self.remove(t,True)
        elif name=='Trap Hole':
            for t in targets:
                c=self.get(t)
                if c and faceup(c) and number(c.get('attack'))>=1000: self.remove(t)
        elif name in ('Man-Eater Bug','Mystical Space Typhoon'):
            for t in targets:
                c=self.get(t)
                if name=='Mystical Space Typhoon' and c.get('name')=='Call of the Haunted':
                    # Only recorded equip/target relationships are known.
                    for bound in c.get('targets',[]): self.remove(bound)
                self.remove(t)
        elif name=='Mirror Force':
            for c in list(self.field.values()):
                if c.get('zone')=='monsters' and c.get('player')!=owner and 'attack' in c.get('position',''): self.remove(c)
        elif name=='Dark Hole':
            for c in list(self.field.values()):
                if c.get('zone')=='monsters': self.remove(c)
        elif name=='Fissure':
            cs=[c for c in self.field.values() if c.get('zone')=='monsters' and c.get('player')!=owner and faceup(c)]
            if cs:
                low=min(number(c.get('attack')) for c in cs)
                tied=[c for c in cs if number(c.get('attack'))==low]
                if len(tied)==1: self.remove(tied[0])
                elif targets:
                    for t in targets:
                        if any(same_card(c,t) for c in tied): self.remove(t); break
                else: self.unknown.append('Fissure tie chosen by engine')
        elif name in ('Monster Reborn','Call of the Haunted'):
            # Revival placement is native; do not invent a future monster slot.
            self.unknown.append(name+' pending native revival')

    def resource_delta(self):
        b=self.b
        value=lambda c: b.value(c) if c.get('zone')=='monsters' else 900
        total=sum((1 if c.get('player')!=b.seat else -1)*value(c) for c in self.destroyed)
        total+=sum((1 if c.get('player')!=b.seat else -1)*value(c)*.42 for c in self.returned)
        return total

    def battle_value(self):
        b=self.b; a,t=b.battle_pair()
        aa=self.get(a); tt=self.get(t)
        if not aa or 'attack' not in aa.get('position',''): return 0
        # Removing the target leads to replay, not a proven direct hit.
        if t.get('zone')=='monsters' and not tt: return 0
        attack=number(aa.get('attack')); ours=aa.get('player')==b.seat
        if not tt:
            if t.get('zone')=='monsters': return 0
            lp=number((b.foe if ours else b.me).get('lp'),8000)
            return (1 if ours else -1)*(attack*.45+(100000 if attack>=lp else 0))
        defense='defense' in tt.get('position','')
        stat=number(tt.get('defense') if defense else tt.get('attack'))
        # Unknown facedown defenders have unknown battle outcome.
        if not tt.get('code'): return 0
        diff=attack-stat; v=0
        if diff>0:
            v=b.value(tt)+(0 if defense else diff*.45)
            if not defense and diff>=number((b.foe if ours else b.me).get('lp'),8000): v+=100000
        elif diff==0 and not defense: v=b.value(tt)-b.value(aa)
        elif diff<0:
            v=-(-diff*.45+(0 if defense else b.value(aa)))
            if not defense and -diff>=number((b.me if ours else b.foe).get('lp'),8000): v-=100000
        return v if ours else -v

    def utility(self):
        return self.resource_delta()+self.battle_value()

    def summary(self):
        return {'destroyed':[c.get('name','unknown') for c in self.destroyed],
                'returned':[c.get('name','unknown') for c in self.returned],
                'battle_value':round(self.battle_value(),1),'unknown':self.unknown,
                'field':[{'player':c.get('player'),'zone':c.get('zone'),'slot':c.get('slot'),
                          'name':c.get('name'),'position':c.get('position'),'attack':c.get('attack')}
                         for c in self.field.values()]}


class Harness(BaseHarness):
    def __init__(self,ctx):
        super().__init__(ctx)
        self.transaction=None
        self.audits=[]
        self.exchange_reviewed=set()
        self.transaction_failed=False
        self.advice_targets=[]
        self.max_calls=8

    def effect_options(self,b,c):
        """Evaluate activation AND target as one decision, against pending links."""
        source=b.card(c.get('card')); name=source.get('name','')
        baseline=Projection(b); a,t=b.battle_pair()
        fighting=bool(a.get('zone')=='monsters') and ('Battle' in str(b.o.get('phase')) or 'Damage' in str(b.o.get('phase')))
        base=baseline.utility() if fighting else baseline.resource_delta()
        def projected(targets):
            extra={'card':source,'targets':targets}
            pr=Projection(b,extra)
            gain=(pr.utility() if fighting else pr.resource_delta())-base
            return pr,gain
        options=[]
        def add(score,targets,why):
            options.append({'score':round(score,3),'targets':targets,'reason':why,'effect':name})
        if name=='Rush Recklessly':
            fighter=a if a.get('player')==b.seat else t
            if not fighting or fighter.get('player')!=b.seat or not faceup(fighter) or 'attack' not in fighter.get('position',''):
                return [{'score':-50,'targets':[],'reason':'No friendly face-up attacker in current battle','effect':name}]
            live=baseline.get(fighter)
            if not live or not faceup(live) or 'attack' not in live.get('position',''):
                return [{'score':-50,'targets':[],'reason':'Pending chain already removes or sets fighter','effect':name}]
            pr,gain=projected([fighter])
            # Mere extra damage is not a card of value, except established lethal.
            add(105 if gain>900 else (-20 if gain<10000 else 150),[fighter], 'Pending-chain battle gain '+str(round(gain)))
        elif name=='Book of Moon':
            for card in b.mine+b.enemy:
                if not faceup(card): continue
                pr,gain=projected([card]); score=-25
                # Chain protection and interrupted combat must improve the actual exchange.
                if gain>700: score=94+min(12,gain/2000)
                if b.own and not b.chain and card.get('player')!=b.seat and 'attack' in card.get('position',''):
                    if b.strongest>number(card.get('defense')) and number(card.get('attack'))>=b.strongest>0:
                        score=max(score,68)
                # Never blank one's own useful attack with an unexplained book.
                add(score,[card],'Book changes exchange by '+str(round(gain)))
        elif name=='Compulsory Evacuation Device':
            for card in b.mine+b.enemy:
                pr,gain=projected([card]); own=card.get('player')==b.seat
                score=-20
                if own:
                    # A save is useful only when public pending effects destroy it.
                    if any(same_card(card,x) for x in baseline.destroyed) and gain>650: score=82
                else:
                    if gain>900 and fighting: score=82+min(15,gain/3000)
                    if number(card.get('level'))>=5: score=max(score,78)
                    if b.own and b.mine and not b.chain:
                        score=max(score,52 if b.strongest<=max(number(card.get('attack')),number(card.get('defense'))) else 20)
                    # A normal summoned attacker can be replayed next turn, but
                    # bounce after summon denies its current attack/tribute body.
                    if not b.own and not b.chain and b.threat>b.strongest and number(card.get('attack'))==b.threat:
                        score=max(score,56)
                add(score,[card],'Bounce exchange gain '+str(round(gain)))
        elif name=='Mystical Space Typhoon':
            pending_sources={refkey(x.get('card') or {}) for x in b.chain
                             if (x.get('card') or {}).get('name')!='Call of the Haunted'}
            for card in b.foe.get('spells_traps',[]):
                if refkey(card) in pending_sources: continue
                pr,gain=projected([card])
                score=65 if b.own and (b.mine or b.me.get('hand')) else -5
                if card.get('name')=='Call of the Haunted' and (card.get('targets') or any((x.get('card') or {}).get('name')=='Call of the Haunted' for x in b.chain)): score=88
                # Cash in a targeted MST instead of losing it, but not on an
                # already activated ordinary spell/trap (destruction != negate).
                if any(any(same_card(source,tgt) for tgt in link.get('targets',[])) for link in b.chain): score=max(score,75)
                add(score,[card],'Remove live backrow; source destruction is not negation')
        elif name=='Mirror Force':
            pr,gain=projected([])
            add(110 if gain>750 else -20,[],'Marginal mass-removal gain '+str(round(gain)))
        elif name=='Trap Hole':
            targets=[]
            for e in reversed(b.events):
                if e.get('event') in ('summon','flip summon'):
                    card=b.card(e.get('card'))
                    if card.get('player')!=b.seat and faceup(card) and number(card.get('attack'))>=1000: targets=[card]
                    break
            if targets:
                pr,gain=projected(targets)
                add(92 if gain>700 else -20,targets,'New summon; marginal destruction '+str(round(gain)))
            else:
                # The offered native timing proves summon eligibility; target
                # identity can still be resolved from its following legal menu.
                add(90 if b.enemy and not b.chain else -15,[],'Native summon window')
        else:
            add(super().activation_score(b,c),[], 'Inherited non-reactive card policy')
        if not options: add(-30,[],'No useful valid target')
        return sorted(options,key=lambda x:x['score'],reverse=True)

    def activation_score(self,b,c):
        return self.effect_options(b,c)[0]['score']

    def subset(self,b):
        # Bind only the active unresolved own effect; other prompts interrupt it.
        tx=self.transaction
        effect=b.effect() or self.last_effect
        if tx and effect==tx['effect'] and b.o.get('round')==tx['round']:
            picks=[]
            for wanted in tx['targets']:
                matches=[i for i,raw in zip(ids_of(b.p),b.p.get('candidates',[]))
                         if same_card(b.card(obj(raw).get('card',raw)),wanted)]
                if matches: picks.append(matches[0])
            if picks and len(picks)==len(tx['targets']):
                action='select %s %s'%(b.p.get('token'),','.join(map(str,picks)))
                if valid(action,b.p): return action
            if tx['targets'] and not tx.get('failed'):
                tx['failed']=True
                self.transaction_failed=True
                self.failures.append('Intended '+tx['effect']+' target no longer offered; use fresh legal menu')
        return super().subset(b)

    def record_action(self,b,action):
        chosen=next((c for c in b.choices if c.get('action')==action),{})
        if 'activat' in str(chosen.get('label','')).lower():
            options=self.effect_options(b,chosen); best=options[0]
            self.transaction={'effect':best['effect'],'targets':best['targets'],
                'round':b.o.get('round'),'turn':b.o.get('turn'),'source':b.card(chosen.get('card')),
                'reason':best['reason'],'start_token':b.p.get('token')}
            # Astra can advise exact CURRENT target locations, never future slots.
            if self.advice_targets:
                advised=[b.card(x) for x in self.advice_targets if isinstance(x,dict)]
                possible=[x for x in options if len(x['targets'])==len(advised) and all(any(same_card(t,a) for a in advised) for t in x['targets'])]
                if possible: self.transaction['targets']=possible[0]['targets']
            self.advice_targets=[]
            self.ctx.log('effect_transaction',**self.transaction)
        if action.startswith('select ') and self.transaction and (b.effect() or self.last_effect)==self.transaction['effect']:
            selected=[]
            try:
                ids=[] if action.split()[2]=='-' else list(map(int,action.split()[2].split(',')))
                selected=[b.card(obj(raw).get('card',raw)) for i,raw in zip(ids_of(b.p),b.p.get('candidates',[])) if i in ids]
            except Exception: pass
            self.transaction['selected']=selected
        # Include the action being submitted, not just links which preceded it.
        # A target selection may resolve the entire chain without another
        # response prompt, so build its prediction before posting the selection.
        if b.chain or 'activat' in str(chosen.get('label','')).lower():
            projected_board=Board(b.o)
            projected_board.chain=[dict(link) for link in b.chain]
            if action.startswith('select ') and self.transaction and self.transaction.get('selected'):
                for link in reversed(projected_board.chain):
                    source=link.get('card') or {}
                    if source.get('player')==b.seat and source.get('name')==self.transaction['effect']:
                        link['targets']=self.transaction['selected']; break
            extra=None
            if 'activat' in str(chosen.get('label','')).lower() and self.transaction:
                extra={'card':self.transaction['source'],'targets':self.transaction['targets']}
            pr=Projection(projected_board,extra)
            links=projected_board.chain+([extra] if extra else [])
            touched=[t for link in links for t in link.get('targets',[])]
            checks=[]
            for t in touched:
                original=b.card(t); expected=pr.get(t)
                if original.get('zone')=='monsters' and original.get('code'):
                    checks.append({'original':original,'expected':dict(expected)})
            self.audits=checks

    def verify_result(self,b,nb,action):
        if b.o.get('round')!=nb.o.get('round'):
            self.transaction=None; self.audits=[]; return
        # Inspect the immediate target menu before spending further resources.
        if self.transaction and nb.p.get('mode')=='subset' and (nb.effect() or self.last_effect)==self.transaction['effect']:
            tx=self.transaction
            if tx['targets'] and any(not any(same_card(nb.card(obj(raw).get('card',raw)),t) for raw in nb.p.get('candidates',[])) for t in tx['targets']):
                tx['failed']=True; self.transaction_failed=True
                self.failures.append('Activation '+tx['effect']+' exposed an unexpected target menu')
        if b.chain and not nb.chain:
            # If an entire battle/turn ran without an intervening prompt, later
            # destruction is not evidence that the earlier effect failed.
            same_window=b.o.get('turn')==nb.o.get('turn') and b.o.get('phase')==nb.o.get('phase')
            mismatches=[]
            if same_window:
                for check in self.audits:
                    old=check['original']; exp=check['expected']
                    # Face-down cards have no stable public identity.  A
                    # source/target that was unknown when the chain was built
                    # cannot provide evidence of a failed consequence.
                    if not old.get('code'):
                        continue
                    # Slots are locations, not identities. A bounced monster can
                    # be immediately Set into the same slot before the next
                    # observation, with its identity hidden again.
                    actual=next((c for c in nb.mine+nb.enemy if same_card(c,old)
                                 and c.get('code')==old.get('code')),None)
                    if exp and actual and exp.get('position')!=actual.get('position'):
                        mismatches.append({'card':old.get('name'),'expected_position':exp.get('position'),'actual_position':actual.get('position')})
                    if not exp and actual:
                        mismatches.append({'card':old.get('name'),'expected':'left field','actual':'still present'})
            if mismatches:
                self.transaction_failed=True
                self.failures.append('Chain consequence mismatch '+dump(mismatches))
                self.ctx.log('transaction_mismatch',checks=mismatches)
            elif self.audits:
                self.ctx.log('transaction_audit',observed=same_window,checks=len(self.audits))
            self.audits=[]; self.transaction=None

    def review_reason(self,b):
        key=(b.o.get('round'),b.o.get('turn'))
        if self.transaction_failed:
            self.transaction_failed=False
            if self.turn_calls.get(key,0)<2: return 'Activation/target/resolution transaction failed; repair against live evidence'
        if self.consequence_failed:
            self.consequence_failed=False
            if self.turn_calls.get(key,0)<2: return 'Position consequence differed from expected destination'
        # Spend deliberation on exchanges, not mandatory placements or every
        # attack window. At most two strategic reviews per duel, plus repairs.
        if self.round_calls.get(b.o.get('round'),0)>=2 or key in self.exchange_reviewed: return None
        if b.main and b.enemy and (b.chain or b.threat>b.strongest or number(b.me.get('lp'))<3000):
            self.exchange_reviewed.add(key)
            return 'Resource disadvantage: find a useful exchange and preserve follow-up'
        if b.p.get('kind')=='chain_response' and len(b.chain)>=2 and any('activat' in c.get('label','') for c in b.choices):
            self.exchange_reviewed.add(key)
            return 'Multiple pending links: verify survival and marginal value of another response'
        return None

    def compact(self,b):
        out={k:v for k,v in b.o.items() if k not in ('card_catalog','recent_events')}
        out['recent_events']=b.events[-16:]
        out['effect_texts']={v['name']:v.get('text','') for v in b.catalog.values() if number(v.get('type'))!=17}
        out['local_action']=self.local(b)
        out['pending_chain_projection']=Projection(b).summary()
        out['activation_target_options']=[{'action':c['action'],'options':self.effect_options(b,c)[:3]}
                 for c in b.choices if 'activat' in str(c.get('label','')).lower()]
        out['active_transaction']=self.transaction
        out['combat_without_new_interaction']=public_combat(b,[c for c in b.mine if 'attack' in c.get('position','')])
        return out

    def ask(self,game,b,reason):
        ctx=self.ctx; remaining=ctx.budget.remaining()
        if self.pending is not None:
            if not self.pending.done(): return None
            try: self.pending.result()
            except Exception: pass
            self.pending=None
        if self.calls>=self.max_calls or remaining.get('system2_calls',0)<=1 or remaining.get('wall_s',0)<260: return None
        wait_allowance=min(210.,240.-(time.monotonic()-self.prompt_started))
        if wait_allowance<125: return None
        self.calls+=1; r=b.o.get('round',1)
        self.round_calls[r]=self.round_calls.get(r,0)+1
        key=(r,b.o.get('turn')); self.turn_calls[key]=self.turn_calls.get(key,0)+1
        self.plan_failed=False
        prompt=(
            'You are a tactical exchange reviewer for this fixed 24-card Yu-Gi-Oh pool. '
            'Choose ONE current legal action. The controller then re-evaluates the new observation; '
            'do not write a future command queue. Code joins effect activation to its target and '
            'projects visible pending chain links in reverse order. The projection is advisory and '
            'does not know hidden cards or simulate future summons. Diagnose any discrepancy. '
            'Prioritize native wins and survival, then card advantage and useful pressure. '
            'Only one Normal Summon OR Set per turn. Keep strong attackers ready for Battle. '
            'The shared label change position / flip summon changes an upright attacker to defense. '
            'Rush Recklessly must target a FACE-UP monster and cannot help a defending monster by '
            'raising its ATK. One pending +700 may already suffice; do not chain a second needlessly. '
            'An already activated ordinary Spell/Trap still resolves if MST destroys its source. '
            'Book of Moon can stop an attacker, protect from face-up-only removal, or let you beat '
            'low DEF; target the actual relevant monster. Do not spend interaction on a harmless attack. '
            'Use current stats, chain targets, and pinned card text. Unknown identities stay unknown. '
            'Newly set quick-play Spells cannot activate that turn. Native menus are authoritative.\n'
            'Return JSON only, under 140 words: {"action":"exact current choose/select/order string",'
            '"targets":[{"player":0,"zone":"monsters","slot":0}],"note":"brief exchange reason"}. '
            'targets is optional advice ONLY when action activates a targeting effect; use current '
            'public locations. Never invent future tokens. For subsets obey count/tribute weights.\n'
            'Trigger: '+reason+'\nFailure evidence: '+dump(self.failures[-5:])+
            '\nProgress: '+dump(ctx.progress())+'\nState: '+dump(self.compact(b)))
        try:
            started=time.monotonic(); clock_started=ctx.clock.now()
            self.pending=ctx.spawn(lambda:ctx.think(prompt,session_id=None,timeout_s=235.))
            while not self.pending.done():
                if ctx.done: return None
                if max(time.monotonic()-started,ctx.clock.now()-clock_started)>wait_allowance:
                    self.failures.append('Planner exceeded 210-second guard; resume live policy')
                    ctx.log('planner_wait_guard',token=b.p.get('token')); return None
                time.sleep(.025)
            result=self.pending.result(); self.pending=None
            parsed=parse_json(result.text)
            action=valid(str(parsed.get('action','')),b.p) or exact_action(result.text,b.p)
            advised=parsed.get('targets',[])
            self.advice_targets=advised[:2] if isinstance(advised,list) else []
            self.plan=[]
            ctx.memory['exchange_note']=str(parsed.get('note',''))[:500]
            ctx.log('exchange_review',reason=reason,action=action,note=ctx.memory['exchange_note'])
            self.failures=[]
            return action
        except (BudgetExceeded,ModelUnavailable): return None
        except Exception as exc:
            self.failures.append('Model recovery '+type(exc).__name__)
            ctx.log('model_recovery',error=type(exc).__name__)
            return None


def build(ctx):
    return Harness(ctx)
