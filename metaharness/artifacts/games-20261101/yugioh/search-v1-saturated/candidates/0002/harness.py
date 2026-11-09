"""Event-driven Yu-Gi-Oh controller: legal action boundary, tactical arithmetic, sparse advisor."""
from __future__ import annotations
import functools
import itertools
import json
import re

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
TRAPS = {'Trap Hole', 'Mirror Force', 'Compulsory Evacuation Device', 'Call of the Haunted'}
QUICK = {'Mystical Space Typhoon', 'Book of Moon', 'Rush Recklessly'}


def unpack(obs):
    if isinstance(obs, str):
        try: return json.loads(obs)
        except Exception: return {}
    return obs if isinstance(obs, dict) else {}


def num(x, default=0):
    try: return int(x)
    except (ValueError, TypeError): return default


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_effect = ''
        self.attacker = None
        self.attack_target = None
        self.round = None
        self.advised = set()
        self.calls = 0
        self.pending = None
        self.turn_moves = set()
        self.last_turn = None
        self.bad = {}

    def setup(self, obs):
        self.o = unpack(obs)
        self.p = self.o.get('prompt') or {}
        self.seat = self.o.get('agent_seat', 0)
        self.me = next((p for p in self.o.get('players', []) if p.get('player') == self.seat), {})
        self.foe = next((p for p in self.o.get('players', []) if p.get('player') != self.seat), {})
        self.mine = self.me.get('monsters', [])
        self.enemies = self.foe.get('monsters', [])
        self.catalog = self.o.get('card_catalog', {})
        if self.round != self.o.get('round'):
            self.round = self.o.get('round')
            self.last_effect = ''
            self.attacker = None
            self.attack_target = None
            self.bad = {}
        turnkey = (self.round, self.o.get('turn'))
        if turnkey != self.last_turn:
            self.turn_moves.clear()
            self.last_turn = turnkey
        self.choices = self.p.get('choices') or []
        self.choices = [c if isinstance(c, dict) else {'action': c, 'label': c} for c in self.choices]

    def card(self, ref):
        if not isinstance(ref, dict): return {}
        base = self.catalog.get(str(ref.get('code')), {}) if isinstance(self.catalog, dict) else {}
        out = dict(base)
        for side in (self.me, self.foe):
            for zone in ('hand', 'monsters', 'spells_traps', 'graveyard', 'banished'):
                for c in side.get(zone, []):
                    if c.get('player') == ref.get('player') and c.get('zone') == ref.get('zone') and c.get('slot') == ref.get('slot'):
                        out.update(c)
        out.update(ref)
        return out

    def value(self, ref):
        c = self.card(ref)
        if num(c.get('type')) & 1 or c.get('zone') == 'monsters':
            return 1000 + max(num(c.get('attack')), num(c.get('defense')) * .65) + (500 if c.get('name') in FLIPS else 0)
        return {'Monster Reborn': 2600, 'Dark Hole': 2200, 'Mirror Force': 2300, 'Fissure': 1800}.get(c.get('name'), 1300)

    def battle_value(self, attacker, target):
        a = num(self.card(attacker).get('attack'))
        if target is None: return a + (20000 if a >= num(self.foe.get('lp'),8000) else 0)
        t = self.card(target)
        if t.get('name') == 'unknown card' or 'attack' not in t:
            return 900 if a >= 2000 else 450 if a >= 1700 else -1000
        if 'defense' in t.get('position', ''):
            d = num(t.get('defense'))
            return self.value(t)*.7 if a > d else -max(200, d-a)
        d = num(t.get('attack'))
        if a > d: return self.value(t)*.7 + a-d + (20000 if a-d >= num(self.foe.get('lp'),8000) else 0)
        if a == d: return (self.value(t)-self.value(attacker))*.7 - 100
        return -self.value(attacker)-(d-a)

    def attack_plan(self):
        """Small public-board assignment search; unknown defenders use an explicit estimate."""
        actions = [c for c in self.choices if c.get('label') == 'attack']
        if not actions: return None
        attackers = [self.card(c.get('card',{})) for c in actions]
        defenders = list(self.enemies)
        @functools.lru_cache(None)
        def search(amask, dmask):
            best = (0.0, None, None)
            for i,a in enumerate(attackers):
                if not amask & (1<<i): continue
                atk = num(a.get('attack'))
                if not dmask:
                    score = atk + search(amask ^ (1<<i),0)[0]
                    if score > best[0]: best = (score,i,None)
                    continue
                for j,t in enumerate(defenders):
                    if not dmask & (1<<j): continue
                    unknown = t.get('name') == 'unknown card' or 'attack' not in t
                    defense = 'defense' in t.get('position','') or unknown
                    strength = 1600 if unknown else num(t.get('defense' if defense else 'attack'))
                    if atk < strength: continue
                    destroyed = atk > strength or (not defense and atk == strength)
                    reward = (self.value(t)*.7 if destroyed else 0) + (max(0,atk-strength) if not defense else 0)
                    if not defense and atk == strength: reward -= self.value(a)*.7
                    # An unknown defender can survive or produce a flip effect.
                    if unknown: reward *= .55
                    if not destroyed: continue
                    value = reward + search(amask ^ (1<<i),dmask ^ (1<<j))[0]
                    if value > best[0]: best = (value,i,j)
            return best
        value,i,j=search((1<<len(attackers))-1,(1<<len(defenders))-1)
        if i is None: return None
        return actions[i]['action'], (defenders[j] if j is not None else None),value

    def active_battle(self):
        # The latest declaration is usable only in the current Battle phase.
        if 'battle' not in str(self.o.get('phase','')).lower() and 'damage' not in str(self.o.get('phase','')).lower(): return None
        for e in reversed(self.o.get('recent_events', [])):
            if e.get('event') == 'attack': return e
            if e.get('event') in ('phase','new_turn'): break
        return None

    def activation_value(self, name):
        own = self.o.get('turn_player') == self.seat
        chain = self.o.get('chain', [])
        enemy_attacks = [m for m in self.enemies if 'attack' in m.get('position','')]
        best_own = max([num(m.get('attack')) for m in self.mine] + [0])
        enemy_peak = max([num(m.get('attack')) for m in self.enemies] + [0])
        if name == 'Trap Hole': return 800 if self.enemies else -100
        if name == 'Mirror Force': return 900 + 300*len(enemy_attacks) if enemy_attacks else -100
        if name == 'Monster Reborn' or name == 'Call of the Haunted':
            graves = self.me.get('graveyard', []) + (self.foe.get('graveyard',[]) if name == 'Monster Reborn' else [])
            power = max([num(m.get('attack')) for m in graves if num(m.get('type'))&1] + [0])
            return 700 + power/10 if power >= 1700 and len(self.mine)<5 else 80 if power and not self.mine else -200
        if name == 'Fissure': return 650 if any('face-up' in m.get('position','') for m in self.enemies) else -200
        if name == 'Dark Hole':
            advantage = sum(self.value(m) for m in self.enemies) - sum(self.value(m) for m in self.mine)
            return 700 + advantage/10 if advantage > 800 else -200
        if name == 'Mystical Space Typhoon':
            backrow = self.foe.get('spells_traps', [])
            if not backrow: return -300
            # Destruction is not negation: usually clear backrow before committing.
            return 420 if own and not chain else 80 if not chain and self.o.get('phase') == 'End' else -50
        if name == 'Compulsory Evacuation Device':
            if not self.enemies: return -200
            if chain and not own: return 100 if enemy_peak >= 2500 else -50
            return 500 if enemy_peak >= 2500 or (own and self.mine) else 100 if not own and self.active_battle() else -100
        if name == 'Book of Moon':
            if own and self.mine and any('attack' in m.get('position','') and num(m.get('defense'))<best_own<=num(m.get('attack')) for m in self.enemies): return 400
            return 180 if not own and self.active_battle() and enemy_peak>best_own else -150
        if name == 'Rush Recklessly':
            combat=self.active_battle()
            if not combat: return -200
            attacker=self.card(combat.get('attacker',{})); target=self.card(combat.get('target',{}))
            a=num(attacker.get('attack'))
            if attacker.get('player')==self.seat:
                if target.get('zone')!='monsters': return 350 if a < num(self.foe.get('lp'),8000) <= a+700 else -150
                d=num(target.get('defense' if 'defense' in target.get('position','') else 'attack'))
                return 500 if a<=d<a+700 else -150
            if target.get('player')==self.seat and 'attack' in target.get('position',''):
                d=num(target.get('attack'))
                return 600 if d<=a<d+700 else -150
            return -150
        if name in FLIPS: return 600
        return -20

    def score_choice(self, ch):
        label = str(ch.get('label','')).lower()
        c = self.card(ch.get('card', {})); name = c.get('name','')
        atk = num(c.get('attack')); defense = num(c.get('defense'))
        kind = self.p.get('kind','')
        if 'shuffle' in label: return -10000
        if 'cancel' in label or 'decline' in label or 'do not' in label or 'no chain' in label or label == 'no': return 0
        if 'end turn' in label: return 0
        if 'main phase 2' in label: return 0
        if 'battle phase' in label: return 20 if any('attack' in m.get('position','') for m in self.mine) else -5
        if 'set spell' in label:
            if name in QUICK and self.o.get('phase') == 'Main 1' and any('battle phase' in str(x.get('label','')).lower() for x in self.choices): return -10
            return 70 if name in TRAPS|QUICK and len(self.me.get('spells_traps',[]))<4 else -20
        if 'normal summon' in label:
            tributes = 0 if num(c.get('level'))<=4 else 1 if num(c.get('level'))<=6 else 2
            sacrifice = sorted(self.value(m) for m in self.mine)[:tributes]
            if tributes and len(sacrifice)<tributes: return -1000
            power = max([num(m.get('attack')) for m in self.enemies if 'attack' in m.get('position','')] + [0])
            score = 200 + atk/10 - sum(sacrifice)*.13
            if name in FLIPS or atk<1400: score -= 400
            if tributes and atk>power>=max([num(m.get('attack')) for m in self.mine]+[0]): score += 400
            if tributes and sum(num(m.get('attack')) for m in self.mine)>atk and not self.enemies: score -= 600
            return score
        if 'set monster' in label:
            if num(c.get('level'))>4: return -100
            score = 100 + defense/20
            if name == 'Man-Eater Bug': score += 230 if self.enemies else 30
            if name == 'Penguin Soldier': score += 230 if self.enemies else 30
            if name == 'Magician of Faith': score += 180 if any(num(g.get('type'))&2 for g in self.me.get('graveyard',[])) else -120
            if atk>=1700: score -= 180
            return score
        if 'flip summon' in label and 'face-down' in c.get('position',''):
            if name == 'Magician of Faith': return 350 if any(num(g.get('type'))&2 for g in self.me.get('graveyard',[])) else -50
            return 350 if name in FLIPS and self.enemies else 180 if atk>=1700 or not self.enemies else -10
        if 'change position' in label or 'change battle position' in label:
            key=(c.get('slot'),name)
            if key in self.turn_moves: return -200
            if 'defense' in c.get('position',''):
                return 160 if atk>=1700 and (not self.enemies or max(self.battle_value(c,t) for t in self.enemies)>0) else -50
            peak=max([num(m.get('attack')) for m in self.enemies]+[0])
            return 110 if defense>atk and atk<peak else -50
        if 'activat' in label: return self.activation_value(name)
        if 'attack' in label and c:
            targets = self.enemies or [None]
            return max(self.battle_value(c,t) for t in targets)
        if kind in ('position','select_position') or ('face-up' in label and ('attack' in label or 'defense' in label)):
            return 20 if 'attack' in label else 10
        if label in ('yes','activate effect'):
            if name == 'Penguin Soldier': return 100 if self.enemies else -10
            return 100 if name in FLIPS else 10
        if kind == 'chain': return self.activation_value(name) if name else -10
        if kind in ('select_card','card','cards','tribute') or ch.get('card'):
            return self.target_value(ch)
        return -10

    def target_value(self, candidate):
        c = self.card(candidate.get('card', candidate))
        name = self.last_effect
        ours = c.get('player') == self.seat
        hint = str(self.p.get('hint','')).lower()
        if ('attack target' in hint or 'attack_target' in str(self.p.get('kind',''))) and self.attacker:
            if self.attack_target and all(c.get(k)==self.attack_target.get(k) for k in ('player','zone','slot')): return 50000
            return self.battle_value(self.attacker,c)
        if 'tribute' in str(self.p.get('kind','')).lower() or 'tribute' in hint or 'discard' in hint:
            return -self.value(c)
        if name in ('Monster Reborn','Call of the Haunted') or 'graveyard' == c.get('zone'):
            return num(c.get('attack'))+ (300 if num(c.get('attack'))>=2500 else 0) if num(c.get('type'))&1 else self.value(c)
        if name == 'Rush Recklessly':
            combat=self.active_battle() or {}
            for ref in (combat.get('attacker',{}),combat.get('target',{})):
                if ours and all(c.get(k)==ref.get(k) for k in ('player','zone','slot')): return 10000
            return num(c.get('attack'))+1000 if ours else -5000
        if name == 'Book of Moon': return self.value(c)+max(0,num(c.get('attack'))-num(c.get('defense'))) if not ours else -5000
        if c.get('zone')=='monsters' and not ours: return self.value(c)
        if c.get('zone')=='spells_traps': return self.value(c) if not ours else -5000
        return -self.value(c) if ours else self.value(c)

    def subset(self, retry=0):
        p=self.p; candidates=p.get('candidates') or []
        cons=p.get('constraint') or p.get('constraints') or {}
        if not isinstance(cons,dict): cons={}
        lo=num(p.get('min',p.get('minimum',cons.get('min',cons.get('minimum',1)))),1)
        hi=num(p.get('max',p.get('maximum',cons.get('max',cons.get('maximum',lo)))),lo)
        kind=str(p.get('kind','')).lower()
        indices=[num(c.get('index'),i) if isinstance(c,dict) else i for i,c in enumerate(candidates)]
        if 'order' in kind or 'sort' in kind:
            return 'order '+str(p.get('token'))+' '+','.join(map(str,indices))
        scores=[self.target_value(c if isinstance(c,dict) else {}) for c in candidates]
        weighted='tribute' in kind or 'sum(weights)' in str(p.get('constraint','')) or any(isinstance(c,dict) and 'weight' in c for c in candidates)
        weights=p.get('weights') or [c.get('weight',1) if isinstance(c,dict) else 1 for c in candidates]
        legal=[]
        if len(candidates)<=20:
            for count in range(0,min(hi,len(candidates))+1):
                if not weighted and count<lo: continue
                for combo in itertools.combinations(range(len(candidates)),count):
                    if weighted and sum(num(weights[i],1) for i in combo)<lo: continue
                    score=sum(scores[i] for i in combo)
                    legal.append((score,combo))
        if legal:
            legal.sort(key=lambda x:x[0],reverse=True)
            picked=legal[retry % len(legal)][1]
        else: picked=sorted(range(len(candidates)),key=lambda i:scores[i],reverse=True)[:min(lo,len(candidates))]
        return 'select '+str(p.get('token'))+' '+(','.join(str(indices[i]) for i in picked) or '-')

    def parse(self, text):
        # Never strip opaque prompt tokens, never synthesize a choose response.
        for ch in self.choices:
            action=ch.get('action','')
            if action and re.search(r'(?<!\w)'+re.escape(action)+r'(?![\w,])',str(text)):
                return action
        return None

    def advise(self, ranked):
        kind=self.p.get('kind','')
        key=(self.round,self.o.get('turn'),kind)
        # At most one consultation per own main phase and 3 per duel.
        if self.pending is not None and not self.pending.done(): return None
        if kind!='main_command' or key in self.advised or self.calls>=9: return None
        if sum(1 for k in self.advised if k[0]==self.round)>=3: return None
        meaningful=[c for _,c in ranked if any(s in str(c.get('label','')).lower() for s in ('activate','normal summon','set monster'))]
        if len(meaningful)<2 or not self.enemies: return None
        rem=self.ctx.budget.remaining()
        if rem.get('system2_calls',0)<=1 or rem.get('wall_s',0)<220: return None
        self.advised.add(key); self.calls+=1
        data={k:self.o.get(k) for k in ('round','your_deck','agent_seat','turn_player','turn','phase','players','chain')}
        data['recent_events']=self.o.get('recent_events',[])[-8:]
        data['prompt']=self.p
        data['heuristic_top']=[{'score':round(s,1),'action':c.get('action')} for s,c in ranked[:5]]
        names={c.get('name') for side in (self.me,self.foe) for z in ('hand','monsters','spells_traps') for c in side.get(z,[]) }
        data['card_texts']={v.get('name'):v.get('text') for v in self.catalog.values() if isinstance(v,dict) and v.get('name') in names and not num(v.get('type'))&16}
        prompt=('Choose the best SINGLE current legal action for this Yu-Gi-Oh duel against a random legal opponent. '
                'Return only the full action string from prompt.choices; preserve token and index. '
                'Prioritize winning and useful resources; use removal to enable attacks, avoid bad tributes, '
                'preserve flip value, avoid destroying your own cards. Destroying a source is not negation. '
                'Heuristic scores are advisory only. Unknown cards are unknown.\n'+json.dumps(data,separators=(',',':')))
        try:
            self.pending = self.ctx.spawn(lambda: self.ctx.think(prompt,timeout_s=min(100,rem['wall_s']-90)))
            # A separate hard wait includes backend queueing, not only generation.
            result=self.pending.result(timeout=min(80,rem['wall_s']-90))
            return self.parse(result.text)
        except Exception as exc:
            self.ctx.log('advisor_failed',error=str(exc)[:180])
            return None

    def run_episode(self, game):
        obs=self.ctx.observe()
        repeated=0; last_token=None
        while not self.ctx.done:
            self.setup(obs)
            if not self.p:
                if getattr(game,'done',False): break
                obs=self.ctx.act('noop'); continue
            token=self.p.get('token')
            repeated=repeated+1 if token==last_token else 0
            last_token=token
            ranked=sorted(((self.score_choice(c),c) for c in self.choices),key=lambda z:z[0],reverse=True)
            if self.p.get('mode')=='subset' or (self.p.get('candidates') and not self.choices):
                action=self.subset(repeated)
            elif self.choices:
                action=ranked[0][1].get('action')
                plan=self.attack_plan() if self.p.get('kind')=='battle_command' else None
                if plan and (ranked[0][1].get('label')=='attack' or ranked[0][0]<=0):
                    action,self.attack_target,_=plan
                if not repeated:
                    action=self.advise(ranked) or action
                else:
                    bad=self.bad.setdefault(token,set()); bad.add(self.ctx.memory.get('last_action'))
                    action=next((c['action'] for _,c in ranked if c.get('action') not in bad),ranked[repeated%len(ranked)][1]['action'])
            else: action=self.subset(repeated)
            chosen=next((c for c in self.choices if c.get('action')==action),{})
            label=str(chosen.get('label','')).lower()
            if 'activat' in label or (label=='yes' and self.p.get('kind')=='activate_effect'):
                self.last_effect=(chosen.get('card') or {}).get('name','')
            if label == 'attack':
                self.attacker = self.card(chosen.get('card') or {})
            if 'change' in label and 'position' in label:
                c=chosen.get('card') or {}; self.turn_moves.add((c.get('slot'),c.get('name','')))
            self.ctx.memory['last_action']=action
            self.ctx.log('policy',kind=self.p.get('kind'),action=action,effect=self.last_effect)
            try: obs=self.ctx.act(action)
            except Exception as exc:
                if self.ctx.done: break
                self.ctx.log('act_failed',error=str(exc)[:180])
                obs=self.ctx.observe()
            if obs is None and not self.ctx.done: obs=self.ctx.observe()


def build(ctx):
    return Harness(ctx)
