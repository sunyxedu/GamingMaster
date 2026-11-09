"""Event driven NetHack controller.

The map, routes, prompts and hard safety gates live in code.  Astra is an
infrequent tactical adviser; ordinary turns never wait for a model answer.
"""
from collections import Counter, deque
import json
import re

DIRS = {'n': (0,-1), 'e': (1,0), 's': (0,1), 'w': (-1,0),
        'ne': (1,-1), 'se': (1,1), 'sw': (-1,1), 'nw': (-1,-1)}
RUN = {'n':'run n','e':'run e','s':'run s','w':'run w',
       'ne':'run ne','se':'run se','sw':'run sw','nw':'run nw'}
FOOD = ('food ration','cram ration','lembas','pancake','apple','orange','pear',
        'melon','banana','carrot','slime mold','fortune cookie','cream pie',
        'candy bar','clove of garlic','sprig of wolfsbane')

def rows(v):
    if isinstance(v, str): return v.splitlines()
    if not isinstance(v, (list, tuple)): return []
    out=[]
    for r in v:
        if isinstance(r, str): out.append(r)
        elif isinstance(r, (list, tuple)):
            out.append(''.join(chr(int(c)) if isinstance(c,(int,float)) and 0 < int(c) < 256 else ' ' for c in r))
    return out

def num(v, d=0):
    try: return int(v)
    except (TypeError, ValueError, OverflowError): return d

def parse_reply(text, names):
    """Parse only a small JSON vocabulary; garbage is a safe empty answer."""
    s=str(getattr(text, 'text', text) or '').strip(); obj={}
    try:
        i=s.find('{')
        if i >= 0: obj=json.JSONDecoder().raw_decode(s[i:])[0]
    except (ValueError, TypeError): obj={}
    if not isinstance(obj, dict): obj={}
    mode=obj.get('mode') if isinstance(obj.get('mode'), str) else ''
    if mode not in ('fight','retreat','explore','descend','rest','food'): mode=''
    act=obj.get('action')
    if not isinstance(act,str) or act not in names: act=None
    target=obj.get('target')
    if not (isinstance(target,(list,tuple)) and len(target)==2 and all(isinstance(x,(int,float)) for x in target)):
        target=None
    run=obj.get('run') if isinstance(obj.get('run'),str) else None
    if run not in DIRS: run=None
    return {'mode':mode,'action':act,'target':target,'run':run,
            'note':str(obj.get('note',''))[:240]}

class Level:
    def __init__(self):
        self.terrain={}; self.visits=Counter(); self.searched=Counter()
        self.blocked=Counter(); self.doors=Counter(); self.stairs={}; self.probed=Counter()

class Harness:
    def __init__(self,ctx):
        self.ctx=ctx; self.levels={}; self.steps=0; self.names=[]
        self.obs={}; self.last=None; self.history=deque(maxlen=12)
        self.mode='explore'; self.target=None; self.target_kind=''; self.advice_run=None
        self.fight=set(); self.pending_dir=None; self.pending_food=None
        self.last_eat=-999; self.last_advice=-999; self.calls=0; self.failures=0
        self.stale=0; self.prompt_seen=None; self.no_model=False

    def arr(self,key,p,d=0):
        try: return num(self.obs.get(key,[])[p[1]][p[0]], d)
        except (KeyError,IndexError,TypeError,ValueError): return d

    def update(self,obs):
        self.obs=obs if isinstance(obs,dict) else {}
        self.stats=self.obs.get('stats',{}) if isinstance(self.obs.get('stats',{}),dict) else {}
        self.pos=(num(self.stats.get('x')),num(self.stats.get('y')))
        self.key=(num(self.stats.get('dungeon_number')),num(self.stats.get('dungeon_level',self.stats.get('depth',1))))
        self.level=self.levels.setdefault(self.key,Level())
        rr=rows(self.obs.get('map',self.obs.get('chars',[])))[:21]
        self.map=[r[:79].ljust(79) for r in rr] + [' '*79]*max(0,21-len(rr))
        self.map=self.map[:21]
        self.msg=str(self.obs.get('message','')); self.tty='\n'.join(rows(self.obs.get('tty',self.obs.get('tty_chars',[]))))
        self.hp=num(self.stats.get('hp'),1); self.maxhp=max(1,num(self.stats.get('hp_max'),1))
        self.turn=num(self.stats.get('time')); self.hunger=num(self.stats.get('hunger'),1)
        self.monsters={}; self.hostiles={}
        for y,row in enumerate(self.map):
            for x,c in enumerate(row):
                p=(x,y)
                if c==' ': continue
                sp=self.arr('specials',p); glyph=self.arr('glyphs',p,-1)
                creature=(c.isalpha() or c in ':;&@') and p != self.pos and not (sp & 32)
                if creature:
                    m={'xy':p,'char':c,'glyph':glyph,'pet':bool(sp&8),'color':self.arr('colors',p)}
                    self.monsters[p]=m
                    if not m['pet']: self.hostiles[p]=m
                elif p != self.pos:
                    # Brown -/| and the known doorway glyphs are open doors.
                    t='/' if (c in '|-' and self.arr('colors',p)==3) or glyph in (2371,2372) else c
                    old=self.level.terrain.get(p); self.level.terrain[p]=t
                    if old != t and t in '.#/<>':
                        for e in list(self.level.blocked):
                            if e[1]==p: self.level.blocked.pop(e,None)
                    if c in '<>': self.level.stairs[p]=c
        self.level.terrain.setdefault(self.pos,'.'); self.level.visits[self.pos]+=1
        low=self.msg.lower()
        if 'staircase down' in low or 'stairs leading down' in low: self.level.stairs[self.pos]='>'
        if self.msg and (not self.history or self.history[-1] != self.msg): self.history.append(self.msg)
        if self.last:
            oldkey,oldpos,oldturn,action,oldhp=self.last
            moved=(oldkey != self.key or oldpos != self.pos)
            self.stale=0 if moved or self.turn != oldturn else self.stale+1
            if oldkey != self.key:
                self.target=None; self.mode='explore'; self.fight=set(); self.advice_run=None
            if self.hp < oldhp: self.target=None
            if action.startswith('move ') and oldkey==self.key and not moved:
                d=action[5:]; dx,dy=DIRS.get(d,(0,0)); q=(oldpos[0]+dx,oldpos[1]+dy)
                dynamic=any(w in low for w in ('in the way','swap places','kitten','dog'))
                if not self.prompt_kind() and not dynamic and self.turn==oldturn and q not in self.monsters:
                    self.level.blocked[(oldpos,q)] += 1
                if self.pos==q: self.level.blocked.pop((oldpos,q),None)
                if any(w in low for w in ('door is closed','door is locked','door resists')):
                    self.level.terrain[q]='+'; self.level.doors[q]+=1
            if action=='search' and not self.prompt_kind():
                for dx,dy in DIRS.values(): self.level.searched[(oldpos[0]+dx,oldpos[1]+dy)] += 1

    def prompt_kind(self):
        s=(self.msg+'\n'+self.tty[:300]).lower()
        if '--more--' in s or '(end)' in s or re.search(r'\(\d+ of \d+\)',s): return 'more'
        if 'what direction' in s or 'in what direction' in s: return 'direction'
        if 'what do you want to eat' in s: return 'eat'
        if '[yn' in s or '(y/n)' in s or re.search(r'\[[ynq ]+\]',s): return 'yesno'
        return None

    def food_action(self):
        inv=self.obs.get('inventory',[]); letters=self.obs.get('inventory_letters',[]); best=None
        for i,item in enumerate(inv if isinstance(inv,list) else []):
            s=str(item).lower()
            if not any(f in s for f in FOOD): continue
            try: k=chr(letters[i]) if isinstance(letters[i],int) else str(letters[i])
            except (IndexError,TypeError,ValueError): continue
            if k in 'kljhunby' and (best is None or ('ration' in s and 'ration' not in best[0])): best=(s,k)
        if best: self.pending_food=best[1]; return 'eat'
        return 'more'

    def prompt_action(self):
        kind=self.prompt_kind()
        if not kind: self.prompt_seen=None; return None
        sig=(kind,self.msg)
        self.prompt_seen=sig
        if kind=='more': return 'more'
        if kind=='direction':
            d=self.pending_dir; self.pending_dir=None
            return 'move '+d if d in DIRS else 'wait'
        if kind=='eat':
            k=self.pending_food or self.food_action(); self.pending_food=None
            return {'k':'move n','l':'move e','j':'move s','h':'move w','u':'move ne','n':'move se','b':'move sw','y':'move nw'}.get(k,'more')
        s=(self.msg+' '+self.tty[:180]).lower()
        # A yes answer is only used for explicit safe food; decline attack/unknown corpse prompts.
        if kind=='yesno':
            return 'move nw' if ('continue eating' in s or 'finish eating' in s or ('food' in s and 'eat' in s)) else 'move se'
        return 'wait'

    def passable(self,p):
        t=self.level.terrain.get(p)
        return t is not None and t not in ' |-=+^{}\\0`'

    def neighbors(self,p):
        for d,(dx,dy) in DIRS.items():
            q=(p[0]+dx,p[1]+dy)
            if not self.passable(q) or q in self.hostiles or self.level.blocked[(p,q)]>1: continue
            if dx and dy:
                if self.level.terrain.get(p)=='/' or self.level.terrain.get(q)=='/': continue
                if not self.passable((p[0]+dx,p[1])) and not self.passable((p[0],p[1]+dy)): continue
            yield q,'move '+d

    def paths(self):
        out={self.pos:[]}; q=deque([self.pos])
        while q:
            p=q.popleft()
            for n,a in self.neighbors(p):
                if n not in out: out[n]=out[p]+[a]; q.append(n)
        return out

    def nearest(self,paths, pred):
        vals=[(len(path),p,path) for p,path in paths.items() if pred(p)]
        return min(vals) if vals else None

    def safe_run(self, action):
        if not action or not action.startswith('move '): return None
        d=action[5:]; dx,dy=DIRS.get(d,(0,0)); p=self.pos; count=0
        # A run is safe only through already mapped corridor squares with no nearby foe.
        while count<4:
            q=(p[0]+dx,p[1]+dy)
            if not self.passable(q) or q in self.hostiles: break
            if dx and dy and (self.level.terrain.get(q)=='/' or self.level.terrain.get(p)=='/'): break
            count+=1; p=q
        if count>=3 and not any(max(abs(m[0]-self.pos[0]),abs(m[1]-self.pos[1]))<=1 for m in self.hostiles):
            return RUN[d]
        return action

    def deterministic(self):
        paths=self.paths()
        # Emergency food and regeneration are hard gates.
        if self.hunger>=3 and self.steps-self.last_eat>25 and self.food_action()=='eat':
            self.last_eat=self.steps; return 'eat'
        near=[(p,m) for p,m in self.hostiles.items() if max(abs(p[0]-self.pos[0]),abs(p[1]-self.pos[1]))<=1]
        if near:
            if self.hp < max(4,int(self.maxhp*.38)):
                # Move toward the safest reachable square away from the foe.
                choices=[]
                for p,path in paths.items():
                    if p==self.pos or any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1 for q,_ in near): continue
                    choices.append((min(max(abs(p[0]-q[0]),abs(p[1]-q[1])) for q,_ in near),-len(path),path))
                if choices and min(choices)[2]: return min(choices)[2][0]
                return 'wait'
            p,m=min(near,key=lambda z:max(abs(z[0][0]-self.pos[0]),abs(z[0][1]-self.pos[1])))
            # A fight answer authorizes only the creatures visible at that call;
            # a newly spawned glyph must trigger a fresh tactical decision.
            if m['glyph'] in self.fight:
                d=(p[0]-self.pos[0],p[1]-self.pos[1]); return 'move '+next((k for k,v in DIRS.items() if v==d),'n')
            # Unknown adjacent creatures are not attacked; ask Astra later, retreat now.
            return 'wait'
        if self.level.stairs.get(self.pos)=='>': return 'down'
        gold=self.nearest(paths,lambda p:self.map[p[1]][p[0]]=='$' and p!=self.pos)
        if gold and (gold[0]<=18 or not any(self.level.stairs.get(p)=='>' for p in paths)): return self.safe_run(gold[2][0])
        stairs=self.nearest(paths,lambda p:self.level.stairs.get(p)=='>')
        if stairs: return self.safe_run(stairs[2][0])
        # Target an unexplored adjacent cell from the closest known frontier.
        fronts=[]
        for p,path in paths.items():
            unknown=[]
            for d,(dx,dy) in DIRS.items():
                q=(p[0]+dx,p[1]+dy)
                if q not in self.level.terrain and self.level.probed[(p,q)]<2: unknown.append((d,q))
            if unknown: fronts.append((len(path)+self.level.visits[p]*.2,p,path,unknown))
        if fronts:
            _,p,path,us=min(fronts)
            if p!=self.pos: return self.safe_run(path[0])
            d,q=min(us,key=lambda z:self.level.probed[(p,z[1])]); self.level.probed[(p,q)]+=1
            return 'move '+d
        doors=[(len(path),p,path) for p,path in paths.items() if self.level.terrain.get(p)=='+' and self.level.doors[p]<8 and max(abs(p[0]-self.pos[0]),abs(p[1]-self.pos[1]))==1]
        if doors:
            _,p,path=min(doors); d=(p[0]-self.pos[0],p[1]-self.pos[1]); self.pending_dir=next((k for k,v in DIRS.items() if v==d),'n'); self.level.doors[p]+=1; return 'kick'
        walls=[]
        for p,path in paths.items():
            ws=[q for q in ((p[0]+dx,p[1]+dy) for dx,dy in DIRS.values()) if self.level.terrain.get(q) in ('|','-') and self.level.searched[q]<12]
            if ws: walls.append((len(path)+self.level.searched[ws[0]],p,path))
        if walls:
            _,p,path=min(walls); return self.safe_run(path[0]) if p!=self.pos and path else 'search'
        return 'wait'

    def adviser_prompt(self):
        paths=self.paths(); goals=[]
        for p,path in list(paths.items())[:120]:
            if self.level.stairs.get(p)=='>': goals.append({'kind':'stairs','xy':p,'d':len(path)})
            if self.map[p[1]][p[0]]=='$': goals.append({'kind':'gold','xy':p,'d':len(path)})
        creatures=[m for m in self.monsters.values()]
        return ('You are a tactical adviser for a turn-based NetHack controller. Return ONLY JSON: '
                '{"mode":"fight|retreat|explore|descend|rest|food","target":[x,y],"run":"e","note":""}. '
                'Do not suggest unsupported commands. Unknown creatures and pets must not be attacked. '
                'Choose progress (gold, stairs, exploration) while preserving HP.\nSTATE '+json.dumps({
                    'stats':self.stats,'message':self.msg,'recent':list(self.history)[-5:],
                    'creatures':creatures,'goals':goals[:24],'mode':self.mode,
                    'wall_s':round(self.ctx.budget.remaining().get('wall_s',0))},separators=(',',':'))+
                '\nMAP\n'+'\n'.join('%02d %s'%(i,r.rstrip()) for i,r in enumerate(self.map) if r.strip()))

    def advise(self):
        rem=self.ctx.budget.remaining()
        if self.no_model or rem.get('system2_calls',0)<=1 or rem.get('wall_s',0)<110: return
        # Delay the first call: the archive showed queued opening calls blocking hundreds of seconds.
        if self.last_advice < 0 and self.steps < 24: return
        if self.steps-self.last_advice<180 and not any(max(abs(p[0]-self.pos[0]),abs(p[1]-self.pos[1]))<=1 for p in self.hostiles): return
        self.last_advice=self.steps; self.calls+=1
        try:
            r=self.ctx.think(self.adviser_prompt(),session_id=None,timeout_s=min(180.0,max(1.0,rem['wall_s']-10)))
            a=parse_reply(r.text,self.names); self.mode=a['mode'] or self.mode
            self.target=tuple(map(int,a['target'])) if a['target'] else None
            self.advice_run=a['run']; self.note=a['note']
            if self.mode=='fight': self.fight.update(m['glyph'] for m in self.hostiles.values())
            if a['action']: self.target=None; self.advice_run=None
            self.ctx.log('tactical_advice',call=self.calls,mode=self.mode,target=self.target,note=self.note)
        except Exception as e:
            self.failures+=1; self.ctx.log('tactical_call_failed',error=type(e).__name__)
            if self.failures>=2: self.no_model=True

    def run_episode(self,game):
        self.names=list(getattr(game.action_spec,'names',[]))
        if not self.names: return
        try: obs=self.ctx.observe()
        except Exception: return
        while not self.ctx.done and self.steps<12000:
            self.update(obs)
            if self.obs.get('done') or self.obs.get('truncated'): break
            if self.ctx.budget.remaining().get('wall_s',0)<=1: break
            action=self.prompt_action(); prompted=action is not None
            if not prompted:
                self.advise()
                if self.target is not None:
                    paths=self.paths()
                    if self.target in paths and self.target!=self.pos: action=self.safe_run(paths[self.target][0])
                    else: self.target=None
                if action is None:
                    action=self.deterministic()
                if self.stale>10: self.target=None; self.mode='explore'; self.stale=0
            if action not in self.names: action='wait' if 'wait' in self.names else self.names[0]
            self.last=(self.key,self.pos,self.turn,action,self.hp); self.steps+=1
            try:
                obs=self.ctx.act(action)
                if obs is None and not self.ctx.done: obs=self.ctx.observe()
            except Exception as e:
                self.ctx.log('action_failed',action=action,error=type(e).__name__)
                if self.ctx.done: break
                self.target=None; self.mode='explore'; self.failures+=1
                try: obs=self.ctx.observe()
                except Exception: break
                if self.failures>=6: break

def build(ctx):
    return Harness(ctx)
