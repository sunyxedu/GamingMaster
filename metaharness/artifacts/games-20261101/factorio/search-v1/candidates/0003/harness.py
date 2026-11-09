"""Goal-driven factory controller. Only public observations and ctx actions are used."""
from __future__ import annotations
import json
import math
import re
import time


def as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def parse_json(text):
    text = str(text).strip()
    for i, c in enumerate(text):
        if c == '{':
            try:
                v, _ = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return {}


def items(x):
    if isinstance(x, dict):
        return {k: int(v) for k, v in x.items() if isinstance(v, (int, float))}
    if isinstance(x, list):
        return {v['name']: int(v.get('count', v.get('amount', 0))) for v in x
                if isinstance(v, dict) and 'name' in v}
    return {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.o = {}
        self.plans = []
        self.targets = {}
        self.failures = []
        self.last_action = ''
        self.recipes = {}
        self.calls = 0
        self.horizon = 8
        self.manual_ticks = 0
        self.manual_jobs = 0
        self.busy_trees = set()
        self.pending = {}
        self.stock = {}
        self.ops = []
        self.quota = 100
        self.blocked = {}
        self.model_future = None

    def entities(self):
        return self.o.get('entities', []) or []

    def inv(self, e, role):
        inv = e.get('inventories', e.get('inventory', {}))
        if isinstance(inv, dict):
            return items(inv.get(role, e.get(role, {})))
        return items(e.get(role, {}))

    def pos(self, e):
        p = e.get('position', {})
        if isinstance(p, dict):
            return float(p.get('x', e.get('x', 0))), float(p.get('y', e.get('y', 0)))
        return float(e.get('x', 0)), float(e.get('y', 0))

    def match(self, p):
        for e in self.entities():
            x, y = self.pos(e)
            if e.get('name') == p['item'] and abs(x-p['x']) < .1 and abs(y-p['y']) < .1:
                return e
        return None

    def add(self, s):
        if len(self.ops) < 30:
            self.ops.append(s)
            return True
        return False

    def transfer(self, e, role, item, n, extract=False):
        n = max(0, min(int(n), self.quota))
        if not extract:
            n = min(n, self.stock.get(item, 0))
        if n and self.add(f"{'extract' if extract else 'insert'} {e['id']} {role} {item} {n}"):
            self.quota -= n
            self.stock[item] = self.stock.get(item, 0) + (n if extract else -n)
            return n
        return 0

    def resource(self, name):
        rows = [r for r in self.o.get('resources', []) if len(r) >= 5 and r[1] == name and r[4] > 0]
        return max(rows, key=lambda r:r[4], default=None)

    def ensure(self, name, target, trail=()):
        """Queue a dependency frontier, never spend outputs before they exist."""
        target = max(0, min(300, int(target)))
        have = self.stock.get(name, 0)
        if have >= target:
            return True
        if have + self.pending.get(name, 0) >= target or name in trail or len(trail) > 10:
            return False
        missing = target - have - self.pending.get(name, 0)
        free = max(0, 3500-self.manual_ticks)
        if free < 30 or self.manual_jobs >= 32:
            return False
        if name in ('iron-ore', 'copper-ore', 'stone', 'coal'):
            r = self.resource(name)
            n = min(missing, int(free//120), 20)
            if r and n > 0 and self.add(f'mine_resource {r[0]} {n}'):
                self.manual_ticks += n*120
                self.manual_jobs += 1
                self.pending[name] = self.pending.get(name, 0)+n
            return False
        if name == 'wood':
            for t in self.o.get('trees', []):
                if t[0] not in self.busy_trees and self.add(f'mine_tree {t[0]}'):
                    self.manual_ticks += 120
                    self.manual_jobs += 1
                    self.busy_trees.add(t[0])
                    self.pending[name] = self.pending.get(name, 0)+4
                    break
            return False
        r = self.recipes.get(name)
        if not r or not r.get('enabled', True):
            self.blocked[name] = 'recipe locked or not observed'
            return False
        if r.get('category') == 'smelting':
            # Furnace pair maintenance produces plates; only hand mine if no drill works.
            return False
        ing = r.get('ingredients', [])
        prod = next((p.get('amount', 1) for p in r.get('products', []) if p.get('name') == name), 1)
        n = max(1, min(math.ceil(missing/prod), int(free/max(30,float(r.get('seconds', .5))*60))))
        ready = True
        for v in ing:
            if v.get('type', 'item') != 'item':
                return False
            if not self.ensure(v['name'], n*v['amount'], trail+(name,)):
                ready = False
        if not ready:
            return False
        ticks = max(1, math.ceil(float(r.get('seconds', .5))*60))*n
        if self.manual_jobs < 32 and self.manual_ticks+ticks <= 3500 and self.add(f'handcraft {name} {n}'):
            for v in ing:
                self.stock[v['name']] = self.stock.get(v['name'], 0)-int(v['amount']*n)
            self.manual_ticks += ticks
            self.manual_jobs += 1
            self.pending[name] = self.pending.get(name, 0)+int(prod*n)
        return False

    def clear(self, x, y, w, h, extra=()):
        for row in self.o.get('water_rows', []):
            if y-h/2 < row['y']+1 and y+h/2 > row['y']:
                for a,b in row['runs']:
                    if x-w/2 < b+1 and x+w/2 > a:
                        return False
        for t in self.o.get('trees', []):
            if abs(t[1]-x)<w/2+.8 and abs(t[2]-y)<h/2+.8:
                return False
        for p in list(self.plans)+list(extra):
            if abs(p['x']-x)<w/2+1 and abs(p['y']-y)<h/2+1:
                return False
        return -125<x<125 and -125<y<125

    def bootstrap(self):
        # Three independent direct-output miner/furnace pairs, selected from observed terrain.
        for ore, count in [('iron-ore', 2), ('copper-ore', 1)]:
            patch = next((p for p in self.o.get('patches', []) if p.get('name') == ore), {})
            center = patch.get('center', {'x':0,'y':0})
            rows = [r for r in self.o.get('resources', []) if r[1] == ore]
            rows.sort(key=lambda r:(r[2]-center['x'])**2+(r[3]-center['y'])**2)
            for _ in range(count):
                for r in rows:
                    x,y = math.floor(r[2]),math.floor(r[3])
                    if self.clear(x,y,2,2) and self.clear(x+2,y,2,2):
                        self.plans.extend([{'item':'burner-mining-drill','x':x,'y':y,'dir':4,'ore':ore},
                                           {'item':'stone-furnace','x':x+2,'y':y,'dir':0,'ore':ore}])
                        break

    def begin(self):
        self.ops=[]
        self.stock=items(self.o.get('stock', {}))
        self.quota=100
        self.blocked={}
        self.pending={}
        self.manual_ticks=0
        self.manual_jobs=len(self.o.get('queue',[]))
        self.busy_trees={q.get('id') for q in self.o.get('queue',[]) if q.get('kind')=='mine_tree'}
        for r in self.o.get('enabled_recipes', []):
            self.recipes[r['name']]=r
        for q in self.o.get('queue', []):
            count=int(q.get('count',1))
            ticks=int(q.get('ticks',q.get('remaining',0)))
            self.manual_ticks+=int(q.get('remaining',ticks))+max(0,count-1)*ticks
            kind=q.get('kind')
            if kind=='handcraft':
                name=q.get('recipe',q.get('name'))
                r=self.recipes.get(name,{})
                for p in r.get('products',[]):
                    self.pending[p['name']]=self.pending.get(p['name'],0)+count*p.get('amount',1)
            if kind=='mine_resource':
                r=next((r for r in self.o.get('resources',[]) if r[0]==q.get('id')),None)
                if r:self.pending[r[1]]=self.pending.get(r[1],0)+count
            if kind=='mine_tree':self.pending['wood']=self.pending.get('wood',0)+4

    def maintenance(self):
        # Fuel first; output extraction creates stock for same-batch crafting.
        for e in self.entities():
            if e.get('name') in ('burner-mining-drill','stone-furnace','boiler'):
                fuel=self.inv(e,'fuel').get('coal',0)
                if fuel < 2:
                    self.transfer(e,'fuel','coal',4-fuel)
        for e in self.entities():
            role='result' if e.get('name')=='stone-furnace' else 'output'
            if e.get('name') in ('stone-furnace','assembling-machine-1','assembling-machine-2'):
                for item,n in self.inv(e,role).items():
                    if n>0:self.transfer(e,role,item,min(n,30),True)
        # All science is valuable when consumed. Keep labs supplied automatically.
        for e in self.entities():
            if e.get('name')=='lab':
                for science in ('automation-science-pack','logistic-science-pack'):
                    self.transfer(e,'lab',science,max(0,8-self.inv(e,'lab').get(science,0)))
        research=self.o.get('research',{})
        current=research.get('current')
        if not current:
            choices=[]
            for r in self.o.get('available_research',[]):
                ingredients=r.get('ingredients',[])
                if r.get('trigger') or not ingredients:continue
                names=set(ingredients) if isinstance(ingredients,dict) else {i['name'] for i in ingredients}
                if names <= {'automation-science-pack'}:
                    choices.append(r)
            if choices:
                choices.sort(key=lambda r:(r['name']!='automation',r.get('units',1)))
                self.add('set_research '+choices[0]['name'])
        for p in self.plans:
            e=self.match(p)
            if not e or not p.get('recipe'):continue
            recipe=p['recipe']
            if e.get('recipe')!=recipe:
                self.add(f"set_recipe {e['id']} {recipe}")
            r=self.recipes.get(recipe,{})
            for i in r.get('ingredients',[]):
                if i.get('type','item')=='item':
                    want=min(30,int(i['amount']*8))
                    n=max(0,want-self.inv(e,'input').get(i['name'],0))
                    self.transfer(e,'input',i['name'],n)

    def action(self):
        self.begin()
        self.maintenance()
        if self.stock.get('coal',0)+self.pending.get('coal',0) < 15:
            self.ensure('coal',25)
        # Construction plans are a persistent ordered queue. Outputs must be observed before placement.
        unfinished=0
        for p in self.plans:
            if self.match(p):continue
            unfinished+=1
            item=p['item']
            if self.stock.get(item,0)>0:
                self.add(f"place {item} {p['x']} {p['y']} {p.get('dir',0)}")
                self.stock[item]-=1
            else:
                self.ensure(item,1)
            if unfinished>=2:break
        for item,n in self.targets.items():
            self.ensure(item,n)
        # While waiting for the first layout, prepare reusable infrastructure from observed recipes.
        # These are stock goals only: power placement and connections remain Astra's decision.
        if self.calls and len(self.plans)<=6 and int(self.o.get('turn',0))>=16:
            for item,n in [('lab',1),('offshore-pump',1),('boiler',1),('steam-engine',1),
                           ('small-electric-pole',4),('automation-science-pack',10)]:
                self.ensure(item,n)
        # Automatically feed declared assemblers' ingredient targets through the worker.
        for p in self.plans:
            if p.get('recipe') and self.match(p):
                r=self.recipes.get(p['recipe'],{})
                for i in r.get('ingredients',[]):
                    if i.get('type','item')=='item':self.ensure(i['name'],min(30,int(i['amount']*8)))
        return '; '.join(self.ops) or 'wait'

    def compact(self):
        keep=('turn','remaining_turns','stock','queue','results','production','lab_consumption','score',
              'research','available_research','enabled_recipes','entities','pump_site','bounds')
        d={k:self.o.get(k) for k in keep}
        d['plans']=self.plans
        d['targets']=self.targets
        d['blocked']=self.blocked
        d['last_action']=self.last_action
        d['failures']=self.failures[-8:]
        d['previous_plan_note']=self.ctx.memory.get('last_plan',{}).get('note','')
        snapshot={k:self.o.get(k,{}) for k in ('production','lab_consumption')}
        snapshot['turn']=self.o.get('turn',0)
        snapshot['score']=self.o.get('score',0)
        previous=self.ctx.memory.get('review_snapshot')
        if previous:
            delta={'turns':snapshot['turn']-previous['turn'],
                   'score':snapshot['score']-previous['score']}
            for key in ('production','lab_consumption'):
                delta[key]={k:v-previous.get(key,{}).get(k,0) for k,v in snapshot[key].items()
                            if v!=previous.get(key,{}).get(k,0)}
            d['since_previous_review']=delta
        self.ctx.memory['review_snapshot']=snapshot
        # Water adjacent to the known pump is sufficient for a compact power site.
        ps=self.o.get('pump_site',{})
        py=ps.get('y',0)
        d['water_near_pump']=[r for r in self.o.get('water_rows',[]) if abs(r['y']-py)<15]
        px=ps.get('x',0)
        d['trees_near_pump']=[t for t in self.o.get('trees',[]) if abs(t[1]-px)<20 and abs(t[2]-py)<20]
        return d

    def think(self):
        budget=self.ctx.budget.remaining()
        # Keep one inference in flight, and reserve actual wall time for all remaining turns.
        # Some backend admission waits can outlive think(timeout_s); never block the factory on them.
        reserve=2.2*int(self.o.get('remaining_turns',128))+30
        if budget.get('wall_s',0)<reserve:return False
        if self.model_future is None and (self.calls>=12 or budget.get('system2_calls',0)<2):return False
        prompt='''You direct a Factorio factory via a persistent goal executor. Maximize real science production and lab consumption in 128 turns. Remote transfer range is unlimited; no need for long belts. Coal fuel, furnace extraction, assembler input/output transfers, lab feeding, red-only research selection, and recipe dependency handcrafting happen automatically each turn. The code has built/directs two iron drill->furnace pairs and one copper pair. A shared 100-item transfer budget applies. The worker mines coal/stone/wood and recursively schedules known enabled crafting recipes (no recursive crafting in one action). Plates come from existing furnaces; each yields ~9 per turn. All machine placement needs valid ground and geometry.
Your job: give a compact construction plan and desired stock. First build steam power and a powered lab near the VALIDATED pump_site, then manually craft red packs for automation research, then powered red-science assemblers, more labs and ongoing science. Crafting a lab unlocks red science. Steam-power unlocks after 50 iron plates; electronics after 10 copper plates. Red science recipe is gear+copper. Handcrafted science earns only lab-consumption credit; assembler-made packs also earn production credit. Avoid green until red works. Boilers need coal and connected water, engines need boiler STEAM, poles must actually connect and cover consumers. A full_output boiler with a no_input_fluid engine indicates disconnected steam: check for a missing pipe across a one-tile gap. Verify actual water and steam flow. Build small, complete chains. Do not make a large belt factory. Resource pairs are already planned and maintained by code.
Return ONLY JSON with these optional keys:
"build": [{"item":"boiler","x":0,"y":0,"dir":0}, ...] to APPEND persistent construction orders; use actual coordinates derived from pump_site. 0=N,4=E,8=S,12=W. Center each odd footprint dimension on a half tile and each even dimension on an integer; handle rectangular buildings separately on each axis. Use the validated pump position and direction unchanged. Include recipe on assembling-machine-1 (e.g. "recipe":"automation-science-pack"). These buildings are crafted automatically with prerequisites. Do not repeat already planned buildings. Up to 12 orders per reply.
"targets": {"automation-science-pack":10,...} replaces desired stock quantities, which code produces via handcrafting when missing. Maintain 5-10 manual red packs until assemblers work. Don't target plates/ore: drills handle those. Construction has priority over targets.
"actions": ["exact legal action string",...] optional immediate repair operations, up to 4 batches. IDs must come from current observations. Use for rotate, extract, mine_building, set_research etc. Standard action grammar from game. Never invent IDs. These execute before the goal loop.
"remove": [{"item":"...","x":0,"y":0}] deletes a WRONG pending construction order only; existing buildings stay until explicitly mined.
"turns": integer 3..10, how long to run the goals before asking again.
"note": one sentence about expected progress and main risk.
Keep response under 900 tokens. Current exact state follows. Fields named inventories are authoritative. Fix failed operations based on results. All quotas reset each turn. Use raw actions for repairs unsupported by the executor.\n'''
        try:
            if self.model_future is None:
                self.calls+=1
                request=prompt+json.dumps(self.compact(),separators=(',',':'))
                timeout=min(220,max(30,budget['wall_s']-reserve))
                self.model_future=self.ctx.spawn(lambda: self.ctx.think(request,timeout_s=timeout))
            stop_waiting=time.monotonic()+180
            while not self.model_future.done():
                if (self.ctx.done or time.monotonic()>=stop_waiting or
                        self.ctx.budget.remaining().get('wall_s',0)<reserve):
                    return False
                time.sleep(.2)
            future,self.model_future=self.model_future,None
            r=future.result()
            data=parse_json(r.text)
        except Exception as exc:
            self.ctx.log('planner_failure',error=str(exc)[:200])
            return False
        self.ctx.memory['last_plan']=data
        for p in data.get('remove',[]) if isinstance(data.get('remove',[]),list) else []:
            if isinstance(p,dict):
                self.plans=[q for q in self.plans if not all(q.get(k)==p.get(k) for k in ('item','x','y'))]
        for p in data.get('build',[])[:12] if isinstance(data.get('build',[]),list) else []:
            if not isinstance(p,dict):continue
            try:
                item=str(p['item'])
                x,y=float(p['x']),float(p['y'])
                direction=int(p.get('dir',0))
                if not re.fullmatch(r'[a-z0-9-]+',item) or not (-128<x<128 and -128<y<128) or direction not in (0,4,8,12):continue
                q={'item':item,'x':x,'y':y,'dir':direction}
                if isinstance(p.get('recipe'),str) and re.fullmatch(r'[a-z0-9-]+',p['recipe']):q['recipe']=p['recipe']
                if not any(q['item']==v['item'] and q['x']==v['x'] and q['y']==v['y'] for v in self.plans):self.plans.append(q)
            except Exception:continue
        if isinstance(data.get('targets'),dict):
            self.targets={}
            for k,v in data['targets'].items():
                try:
                    if isinstance(k,str) and re.fullmatch(r'[a-z0-9-]+',k) and isinstance(v,(int,float)) and math.isfinite(v):
                        self.targets[k]=max(0,min(100,int(v)))
                except (ValueError,TypeError,OverflowError):
                    pass
        try:self.horizon=max(3,min(10,int(data.get('turns',8))))
        except Exception:self.horizon=8
        for a in data.get('actions',[])[:4] if isinstance(data.get('actions',[]),list) else []:
            if isinstance(a,str) and len(a)<2500 and not self.ctx.done:self.step(a)
        return bool(data)

    def step(self, action):
        if self.ctx.done:return
        self.last_action=action
        try:
            v=self.ctx.act(action)
            self.o=as_dict(v) if v is not None else as_dict(self.ctx.observe())
        except Exception as exc:
            self.failures.append({'action':action,'exception':str(exc)[:200]})
            if not self.ctx.done:
                try:self.o=as_dict(self.ctx.observe())
                except Exception:return
        for result in self.o.get('results',[]):
            if isinstance(result,dict) and result.get('ok') is False:
                self.failures.append({'turn':self.o.get('turn'),'action':action,'result':result})
        self.failures=self.failures[-20:]

    def run_episode(self, game):
        self.o=as_dict(game.observe())
        if not self.o:return
        self.bootstrap()
        next_think=16
        no_progress=0
        for _ in range(180):
            if self.ctx.done or self.o.get('remaining_turns',1)<=0:break
            turn=self.o.get('turn',0)
            if turn>=next_think:
                self.think()
                if self.ctx.done:break
                next_think=self.o.get('turn',turn)+self.horizon
            old=self.o.get('turn',0)
            self.step(self.action())
            no_progress=no_progress+1 if self.o.get('turn',0)==old else 0
            if no_progress>=2:
                self.step('wait')
                no_progress=0


def build(ctx):
    return Harness(ctx)
