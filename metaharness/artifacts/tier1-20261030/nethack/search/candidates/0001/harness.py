"""Compact observation + feedback-controlled navigation, with Astra as supervisor."""
from __future__ import annotations
import json
import re
from collections import Counter, deque

DIRS = {'n': (0,-1), 'e': (1,0), 's': (0,1), 'w': (-1,0),
        'ne': (1,-1), 'se': (1,1), 'sw': (-1,1), 'nw': (-1,-1)}
KEYS = {'k':'move n','l':'move e','j':'move s','h':'move w','u':'move ne',
        'n':'move se','b':'move sw','y':'move nw','K':'run n','L':'run e',
        'J':'run s','H':'run w','U':'run ne','N':'run se','B':'run sw',
        'Y':'run nw','e':'eat','s':'search','.':'wait','>':'down','<':'up'}
# These glyph/character correspondences are present in the archived observations.
OPEN_DOORS = {2371,2372}
FOODS = ('food ration','cram ration','lembas','pancake','apple','orange','pear',
         'melon','carrot','banana','fortune cookie','slime mold','cream pie')

def text_rows(value):
    if isinstance(value,str): return value.splitlines()
    if not isinstance(value,(list,tuple)): return []
    return [r if isinstance(r,str) else ''.join(chr(int(c)) if int(c) else ' ' for c in r) for r in value]

def parse_reply(text,names):
    result = {}
    try:
        start=text.index('{'); result=json.JSONDecoder().raw_decode(text[start:])[0]
    except (ValueError,TypeError): pass
    if not isinstance(result,dict): result={}
    actions=result.get('actions',[])
    if isinstance(actions,str): actions=[actions]
    if not isinstance(actions,list): actions=[]
    good=[]
    for a in actions[:32]:
        if isinstance(a,int) and 0<=a<len(names): good.append(names[a])
        elif isinstance(a,str) and a.strip().lower() in names: good.append(a.strip().lower())
    if not good and not result:
        # Accept a plain valid action, but never invent an action from arbitrary prose.
        for line in str(text).splitlines():
            a=line.strip(' `\t\r\"\',').lower()
            if a in names: good.append(a)
    result['actions']=good
    return result

class Level:
    def __init__(self):
        self.terrain={}; self.visits=Counter(); self.searched=Counter()
        self.failed=Counter(); self.doors=Counter(); self.down=set(); self.up=set()
        self.last_new=0; self.goal=None

class Harness:
    def __init__(self,ctx):
        self.ctx=ctx; self.levels={}; self.steps=0; self.history=deque(maxlen=16)
        self.queue=deque(); self.mode='explore'; self.target=None
        self.last_call=-100; self.calls=0; self.pending=None; self.last_food=-1000
        self.last_obs=None; self.last_action=None; self.last_pos=None
        self.repeat=0; self.last_sig=None; self.last_danger_call=-100
        self.friendly=set(); self.avoid=set(); self.note=''; self.errors=0
        self.last_heal_hp=0; self.rest_count=0; self.hunger_fail=0

    def setup(self,obs):
        self.obs=obs; self.stats=obs.get('stats',{})
        self.pos=(int(self.stats.get('x',0)),int(self.stats.get('y',0)))
        self.level_key=(self.stats.get('dungeon_number',0),self.stats.get('dungeon_level',self.stats.get('depth',1)))
        self.level=self.levels.setdefault(self.level_key,Level())
        self.rows=text_rows(obs.get('map',obs.get('chars',[])))
        self.h=len(self.rows); self.w=max(map(len,self.rows),default=0)
        self.msg=str(obs.get('message',''))
        self.tty='\n'.join(text_rows(obs.get('tty',obs.get('tty_chars',[]))))
        self.hp=int(self.stats.get('hp',1)); self.maxhp=max(1,int(self.stats.get('hp_max',1)))
        self.turn=int(self.stats.get('time',0)); self.hunger=int(self.stats.get('hunger',1))
        self.monsters=[]; new=0
        for y,row in enumerate(self.rows):
            for x,c in enumerate(row):
                p=(x,y)
                if c==' ': continue
                g=self.array('glyphs',p,-1); special=self.array('specials',p,0)
                monster=(c.isalpha() or c in ':;&@') and p!=self.pos
                if monster:
                    tame=bool(special & 8) or g in self.friendly
                    # Initial tame highlighting is observed, never assume every dog is friendly.
                    if special & 8: self.friendly.add(g)
                    self.monsters.append({'x':x,'y':y,'char':c,'glyph':g,'tame':tame,
                                          'color':self.array('colors',p,0)})
                if p not in self.level.terrain: new+=1
                if c in '.#<>+|-^{}\\' or g in OPEN_DOORS:
                    self.level.terrain[p]=(c,g)
                elif p not in self.level.terrain:
                    self.level.terrain[p]=('.',-1)
                if c=='>': self.level.down.add(p)
                if c=='<': self.level.up.add(p)
        self.level.terrain.setdefault(self.pos,('.',-1))
        if new: self.level.last_new=self.steps
        self.level.visits[self.pos]+=1
        if self.msg and (not self.history or self.msg!=self.history[-1]): self.history.append(self.msg)
        self.hostiles=[m for m in self.monsters if not m['tame'] and m['glyph'] not in self.avoid]
        self.near=[m for m in self.hostiles if self.dist(self.pos,(m['x'],m['y']))<=2]
        sig=(self.level_key,self.pos,self.turn,self.msg)
        self.repeat=self.repeat+1 if sig==self.last_sig else 0; self.last_sig=sig
        if self.last_pos is not None and self.last_action and self.last_action.startswith('move ') and self.last_level==self.level_key:
            d=self.last_action.split()[1]; dx,dy=DIRS[d]
            q=(self.last_pos[0]+dx,self.last_pos[1]+dy)
            if self.pos==self.last_pos and not self.prompt_kind() and self.last_action_kind!='prompt':
                if not any((m['x'],m['y'])==q for m in self.monsters):
                    self.level.failed[(self.pos,q)]+=1
                    if self.level.failed[(self.pos,q)]>=2: self.queue.clear(); self.target=None
            elif self.pos==q: self.level.failed.pop((self.last_pos,q),None)
        if self.last_action=='search' and self.last_action_kind!='prompt':
            for dx,dy in DIRS.values(): self.level.searched[(self.pos[0]+dx,self.pos[1]+dy)]+=1
        if self.last_level != self.level_key if hasattr(self,'last_level') else False:
            self.target=None; self.queue.clear(); self.pending=None

    def array(self,key,p,default):
        try: return int(self.obs[key][p[1]][p[0]])
        except (KeyError,IndexError,TypeError,ValueError): return default

    def char(self,p):
        x,y=p
        return self.rows[y][x] if 0<=y<self.h and 0<=x<len(self.rows[y]) else ' '

    @staticmethod
    def dist(p,q): return max(abs(p[0]-q[0]),abs(p[1]-q[1]))

    def prompt_kind(self):
        low=(self.msg+'\n'+self.tty[:240]).lower()
        if '--more--' in self.tty.lower() or '--more--' in self.msg.lower(): return 'more'
        if 'in what direction' in low or 'what direction' in low: return 'direction'
        if 'what do you want to eat' in low: return 'food'
        if re.search(r'\[[ynq]+\]',low) or '[yn' in low or '(y/n)' in low: return 'yn'
        if '(end)' in self.tty.lower() or re.search(r'\(\d+ of \d+\)',self.tty): return 'more'
        return None

    def food_action(self):
        inv=self.obs.get('inventory',[]); letters=self.obs.get('inventory_letters',[])
        choices=[]
        for i,item in enumerate(inv):
            s=str(item).lower()
            if not any(f in s for f in FOODS): continue
            try:
                k=letters[i]; k=chr(k) if isinstance(k,int) else str(k)
            except (IndexError,TypeError,ValueError): continue
            if k in KEYS and KEYS[k] in self.names: choices.append((0 if 'ration' in s else 1,KEYS[k]))
        return min(choices)[1] if choices else None

    def prompt_action(self):
        kind=self.prompt_kind()
        if kind=='more': return 'more'
        if kind=='direction':
            if self.pending and self.pending in DIRS: return 'move '+self.pending
            for d,(dx,dy) in DIRS.items():
                if self.char((self.pos[0]+dx,self.pos[1]+dy))=='+': return 'move '+d
            return 'wait'
        if kind=='food': return self.food_action() or 'more'
        if kind=='yn':
            s=(self.msg+'\n'+self.tty[:180]).lower()
            yes=('continue eating' in s or 'finish eating' in s)
            if 'eat' in s and any(f in s for f in FOODS): yes=True
            return 'move nw' if yes else 'move se'
        return None

    def passable(self,p,allow_unknown=False):
        if not (1<=p[0]<self.w and 0<=p[1]<self.h): return False
        if p not in self.level.terrain: return allow_unknown
        c,g=self.level.terrain[p]
        return g in OPEN_DOORS or c not in '|-+^{}\\'

    def neighbors(self,p,unknown=False,monsters=True):
        for d,(dx,dy) in DIRS.items():
            q=(p[0]+dx,p[1]+dy)
            if not self.passable(q,unknown): continue
            if self.level.failed[(p,q)]>=2: continue
            if dx and dy:
                if self.level.terrain.get(p,('',0))[1] in OPEN_DOORS or self.level.terrain.get(q,('',0))[1] in OPEN_DOORS: continue
                if not self.passable((p[0]+dx,p[1])) and not self.passable((p[0],p[1]+dy)): continue
            if monsters and any((m['x'],m['y'])==q for m in self.hostiles): continue
            yield q,'move '+d

    def paths(self):
        paths={self.pos:[]}; todo=deque([self.pos])
        while todo:
            p=todo.popleft()
            for q,a in self.neighbors(p):
                if q not in paths:
                    paths[q]=paths[p]+[a]; todo.append(q)
        return paths

    def navigation(self):
        paths=self.paths(); p=self.pos
        if self.target is not None:
            if self.target==p: self.target=None
            elif self.target in paths: return paths[self.target][0]
            else: self.target=None
        # Staircase targets and gold give concrete progress instead of exhaustive room coverage.
        if p in self.level.down and self.mode!='explore_only': return 'down'
        stairs=[q for q in self.level.down if q in paths and paths[q]]
        gold=[q for q in paths if self.char(q)=='$' and paths[q] and self.level.visits[q]<3]
        if gold:
            q=min(gold,key=lambda q:len(paths[q]))
            if len(paths[q])<=12 or not stairs: return paths[q][0]
        if stairs and self.mode!='explore_only':
            return paths[min(stairs,key=lambda q:len(paths[q]))][0]
        options=[]
        for q,path in paths.items():
            unknowns=[r for r,a in self.neighbors(q,unknown=True) if r not in self.level.terrain]
            if unknowns:
                # An unknown cell just beyond a known walkable cell is a frontier.
                options.append((len(path)+2*self.level.visits[q],q,unknowns))
        if options:
            _,q,unknowns=min(options,key=lambda z:z[0])
            if q!=p: return paths[q][0]
            r=min(unknowns,key=lambda r:self.level.failed[(q,r)])
            dx,dy=r[0]-p[0],r[1]-p[1]
            return 'move '+next(d for d,v in DIRS.items() if v==(dx,dy))
        # Doors: approach cardinally, try bump-opening before kicking a confirmed door.
        doors=[]
        for q,(c,g) in self.level.terrain.items():
            if c!='+' or self.level.doors[q]>=12: continue
            for d,(dx,dy) in list(DIRS.items())[:4]:
                near=(q[0]-dx,q[1]-dy)
                if near in paths: doors.append((len(paths[near])+self.level.doors[q],near,q,d))
        if doors:
            _,near,q,d=min(doors,key=lambda z:z[0])
            if near!=p: return paths[near][0]
            self.level.doors[q]+=1
            if self.level.doors[q]==1: return 'move '+d
            self.pending=d; return 'kick'
        # Search each wall area with explicit per-wall coverage; retry probabilistic searches.
        spots=[]
        for q,path in paths.items():
            walls=[]
            for dx,dy in DIRS.values():
                r=(q[0]+dx,q[1]+dy); c,g=self.level.terrain.get(r,(' ',0))
                if (c in '|-' and g not in OPEN_DOORS) or (c==' ' and self.level.terrain.get(q,('',0))[0]=='#'):
                    walls.append(r)
            if walls:
                n=min(self.level.searched[w] for w in walls)
                if n<36: spots.append((len(path)+n*2,q))
        if spots:
            _,q=min(spots)
            return paths[q][0] if q!=p else 'search'
        return None

    def tactical(self):
        adjacent=[m for m in self.hostiles if self.dist(self.pos,(m['x'],m['y']))==1]
        if not adjacent: return None
        if self.hp < max(5,self.maxhp*.45):
            choices=[]
            for p,a in self.neighbors(self.pos):
                distance=min(self.dist(p,(m['x'],m['y'])) for m in self.hostiles)
                choices.append((distance,-self.level.visits[p],p,a))
            if choices:
                best=max(choices)
                if best[0]>1: return best[-1]
        # One attack, then re-observe; do not send an open-loop combat burst.
        m=min(adjacent,key=lambda m:(m['char'] in '@&',m['x'],m['y']))
        delta=(m['x']-self.pos[0],m['y']-self.pos[1])
        return 'move '+next(d for d,v in DIRS.items() if v==delta)

    def ask(self,reason):
        budget=self.ctx.budget.remaining()
        if budget.get('system2_calls',0)<=1 or budget.get('wall_s',0)<100 or self.calls>=18: return False
        inventory=[]
        for i,s in enumerate(self.obs.get('inventory',[])):
            if not s: continue
            letters=self.obs.get('inventory_letters',[])
            k=letters[i] if i<len(letters) else 0
            if isinstance(k,int): k=chr(k) if k else '?'
            inventory.append(str(k)+': '+str(s))
        maptext='   '+''.join(str(x//10 or ' ') if x%10==0 else ' ' for x in range(self.w))+'\n'
        maptext+='   '+''.join(str(x%10) for x in range(self.w))+'\n'
        maptext+='\n'.join('%02d %s'%(y,r.rstrip()) for y,r in enumerate(self.rows) if r.strip())
        prompt=(
          'You supervise a NetHack score agent. Maximize score and survival with only the listed actions. '
          'Code already navigates shortest paths to gold, downstairs and exploration frontiers, opens/kicks doors, '
          'searches hidden exits, handles prompts, fights adjacent hostiles one step at a time and rests when safe. '
          'Choose a strategic correction only if needed. This is a turn game; thinking is free in game turns but costly in wall time. '
          'Treat visible monster letters as uncertain; tame flags come from highlighting. Never attack peaceful humans. '
          'No quaff, wield, pickup, escape, arbitrary letters or extended commands exist in this action set. '
          'Inventory prompt letters can only be typed using corresponding vi action keys: k=move n, l=move e, j=move s, '
          'h=move w, u=move ne, n=move se, b=move sw, y=move nw, e=eat, s=search; uppercase via run. '
          'Return a short JSON object, no prose: {"mode":"explore", "target":null, "actions":[], '
          '"ignore_glyphs":[], "note":"brief durable advice", "stop":false}. '
          'mode is explore (seek stairs), explore_only (stay on level), or rest. target optionally [x,y] for code pathfinding. '
          'actions optionally 1-12 exact action names for an immediate maneuver; code interrupts on damage or failed movement. '
          'ignore_glyphs are observed monster glyph IDs known peaceful, not merely weak. '
          'stop=true only when further play is hopeless or cannot safely gain score. '
          'Prefer empty actions and let navigation run. Give no speculative multi-turn combat scripts.\n'
          'Allowed actions: '+', '.join(self.names)+'\n'
          'Reason: '+reason+'\nStats: '+json.dumps(self.stats)+'\n'
          'Message: '+self.msg+'\nInventory: '+json.dumps(inventory)+'\n'
          'Monsters: '+json.dumps(self.monsters)+'\n'
          'Recent messages: '+json.dumps(list(self.history))+'\n'
          'Prior advice: '+self.note+'\nMode: '+self.mode+'; steps since call: '+str(self.steps-self.last_call)+'\n'
          'ASCII coordinates x=column, y=row, n decreases y. @ is player. + closed door, > downstairs, $ gold, blank unknown.\n'
          +maptext+'\nTTY:\n'+self.tty[:2200])
        self.last_call=self.steps; self.calls+=1
        self.ctx.log('supervisor_request',reason=reason,stats=self.stats,steps=self.steps)
        try:
            answer=self.ctx.think(prompt,session_id=None,timeout_s=min(240.,max(1.,budget['wall_s']-15)))
            reply=parse_reply(answer.text,self.names)
        except Exception as exc:
            self.ctx.log('supervisor_failure',error=type(exc).__name__)
            return False
        self.note=str(reply.get('note',self.note))[:600]
        if reply.get('mode') in ('explore','explore_only','rest'): self.mode=reply['mode']
        target=reply.get('target')
        if isinstance(target,list) and len(target)==2 and all(isinstance(v,int) for v in target): self.target=tuple(target)
        vals=reply.get('ignore_glyphs',[])
        if isinstance(vals,list): self.avoid.update(v for v in vals if isinstance(v,int) and any(m['glyph']==v for m in self.monsters))
        self.queue=deque(reply['actions'])
        self.ctx.log('supervisor_plan',plan=reply)
        return reply.get('stop') is True

    def run_episode(self,game):
        self.names=list(game.action_spec.names)
        obs=self.ctx.observe()
        while not self.ctx.done and self.steps<6000:
            if not isinstance(obs,dict):
                obs=self.ctx.observe()
                if not isinstance(obs,dict): return
            if obs.get('done') or obs.get('truncated'): return
            self.setup(obs)
            action=self.prompt_action(); kind='prompt' if action else 'auto'
            if not action:
                if self.pending: self.pending=None
                danger=bool(self.near) and (self.hp<self.maxhp*.6 or any(m['char'] in '@&' for m in self.near))
                reason=None
                if self.calls==0: reason='opening: choose role-specific strategy'
                elif danger and self.steps-self.last_danger_call>=12:
                    reason='danger: low HP or potentially peaceful/dangerous human nearby'; self.last_danger_call=self.steps
                elif self.repeat>=3: reason='stuck: repeated identical observation'
                elif self.steps-self.last_call>=120: reason='progress review'
                if reason and self.steps-self.last_call>=8:
                    if self.ask(reason): return
                    if self.ctx.done: return
                if self.repeat>8: return
                # Explicit food gate; unavailable inventory letters are not fabricated.
                if self.hunger>=2 and self.turn-self.last_food>20 and self.food_action() and not self.near:
                    action='eat'; self.last_food=self.turn; self.queue.clear()
                if not action and self.queue:
                    action=self.queue.popleft(); kind='model'
                    if action.startswith('run '): action='move '+action.split()[1]
                if not action: action=self.tactical()
                if not action and not self.near and self.hp<self.maxhp*.85 and self.hunger<3 and self.rest_count<160:
                    action='search'; self.rest_count+=1
                else:
                    if self.hp>=self.maxhp*.85: self.rest_count=0
                if not action: action=self.navigation()
                if not action:
                    if self.steps-self.last_call>=8 and self.ask('no reachable frontier or unsearched wall'): return
                    action=self.queue.popleft() if self.queue else None
                    if not action: return
            if action not in self.names: action='search' if 'search' in self.names else self.names[0]
            self.last_action=action; self.last_action_kind=kind; self.last_pos=self.pos; self.last_level=self.level_key
            oldhp=self.hp
            try:
                obs=self.ctx.act(action)
                if obs is None: obs=self.ctx.observe()
            except Exception as exc:
                self.ctx.log('action_failure',action=action,error=type(exc).__name__)
                self.errors+=1
                if self.ctx.done or self.errors>=3: return
                self.queue.clear(); obs=self.ctx.observe()
            self.steps+=1
            if isinstance(obs,dict) and obs.get('stats',{}).get('hp',oldhp)<oldhp:
                self.queue.clear(); self.target=None
            if self.steps%40==0:
                self.ctx.log('checkpoint',steps=self.steps,stats=obs.get('stats',{}) if isinstance(obs,dict) else {},
                             message=obs.get('message','') if isinstance(obs,dict) else '')


def build(ctx):
    return Harness(ctx)
