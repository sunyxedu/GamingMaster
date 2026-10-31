"""Machine/worker specialization: engine green science, manual red supplementation."""
from __future__ import annotations
import json
import math
import re


def obj(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return {}


def inventory(e, role):
    data = e.get('inventories', e.get('inventory', {}))
    v = data.get(role, {}) if isinstance(data, dict) else {}
    if isinstance(v, dict):
        return {k: int(n) for k, n in v.items() if isinstance(n, (int, float))}
    if isinstance(v, list):
        return {r['name']: r.get('count', r.get('amount', 0)) for r in v if isinstance(r, dict) and 'name' in r}
    return {}


def pos(e):
    p = e.get('position', e)
    return float(p.get('x', 0)), float(p.get('y', 0))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ob = {}
        self.recipes = {}
        self.tasks = []
        self.index = 0
        self.furnace_ores = {}
        self.chest_items = {}
        self.repairs = []
        self.failures = []
        self.last_call = -20
        self.calls = 0
        self.blocked = 0
        self.last_index = -1
        self.last_action = ''
        self.advisor_future = None
        self.advisor_goal = None
        self.machine_recipes = {}
        self.green_wait = 0
        self.reserve = {}

    def entity(self, name, x, y):
        for e in self.ob.get('entities', []):
            if e.get('name') == name:
                ex, ey = pos(e)
                if abs(ex-x) < .1 and abs(ey-y) < .1:
                    return e
        return None

    def init_tasks(self):
        # Sites depend exclusively on the current public map.
        self.sites = {}
        for kind in ('iron-ore', 'copper-ore', 'coal', 'stone'):
            rows = [r for r in self.ob.get('resources', []) if r[1] == kind]
            if not rows:
                continue
            cx = sum(r[2] for r in rows)/len(rows)
            cy = sum(r[3] for r in rows)/len(rows)
            self.sites[kind] = sorted(rows, key=lambda r: (r[2]-cx)**2+(r[3]-cy)**2)
        occupied = []
        def site(kind):
            for r in self.sites.get(kind, []):
                x, y = math.floor(r[2]), math.floor(r[3])
                if all(abs(x-a) >= 5 or abs(y-b) >= 3 for a,b in occupied):
                    occupied.append((x,y))
                    return x,y
            return 0, 0
        def pair(kind):
            x,y=site(kind)
            self.tasks.extend([('place','burner-mining-drill',x,y,4), ('place','stone-furnace',x+2,y,0)])
            self.furnace_ores[(x+2,y)] = kind
        pair('iron-ore')
        pair('copper-ore')
        # Expand raw material supply after first science, preserving startup turns.
        x,y=site('iron-ore')
        expansion=[('place','burner-mining-drill',x,y,4), ('place','stone-furnace',x+2,y,0)]
        self.furnace_ores[(x+2,y)]='iron-ore'
        x,y=site('coal')
        expansion.extend([('place','burner-mining-drill',x,y,4), ('place','wooden-chest',x+1.5,y-.5,0)])
        self.chest_items[(x+1.5,y-.5)]='coal'
        # A pump faces water. Rotate this north-coast template with its direction.
        p=self.ob.get('pump_site', {})
        px,py=float(p.get('x',.5)),float(p.get('y',-60.5))
        direction=int(p.get('direction',0))
        def transform(dx,dy):
            for _ in range(direction//4):
                dx,dy=-dy,dx
            return px+dx,py+dy
        def put(item, dx,dy,d=0):
            a,b=transform(dx,dy)
            self.tasks.append(('place',item,a,b,(d+direction)%16))
        self.tasks.append(('place','offshore-pump',px,py,direction))
        put('pipe',0,1)
        put('pipe',0,2)
        put('boiler',.5,4,4)
        put('pipe',2,4)
        put('steam-engine',5,4,4)
        # Poles and consumers kept adjacent to the generator, away from shoreline.
        put('small-electric-pole',3,7)
        put('small-electric-pole',8,7)
        put('lab',3,10)
        self.tasks.append(('science',10))
        self.tasks.append(('research','automation'))
        self.tasks.append(('unlock','assembling-machine-1'))
        self.tasks.extend(expansion)
        put('assembling-machine-1',8,10)
        # A second coal source pays for the larger burner fleet.
        x,y=site('coal')
        self.tasks.extend([('place','burner-mining-drill',x,y,4),
                           ('place','wooden-chest',x+1.5,y-.5,0)])
        pair('iron-ore')
        pair('copper-ore')
        put('small-electric-pole',3,13)
        put('lab',3,16)
        put('small-electric-pole',8,13)
        put('assembling-machine-1',8,16)
        self.machine_recipes[transform(8,16)]='logistic-science-pack'
        pair('iron-ore')
        put('small-electric-pole',3,19)
        put('lab',3,22)
        put('small-electric-pole',8,19)
        put('assembling-machine-1',8,22)
        self.machine_recipes[transform(8,22)]='logistic-science-pack'
        pair('iron-ore')
        for row in (7,13):
            put('small-electric-pole',-2,row)
            put('lab',-2,row+3)
        self.power_origin=(px,py,direction)

    @staticmethod
    def size(name, direction):
        if name in ('burner-mining-drill','stone-furnace'):
            return 2,2
        if name in ('lab','assembling-machine-1'):
            return 3,3
        if name=='boiler':
            return (2,3) if direction%8==4 else (3,2)
        if name=='steam-engine':
            return (5,3) if direction%8==4 else (3,5)
        return 1,1

    def blocking_tree(self, name, x, y, direction):
        if name=='offshore-pump':
            return None  # The observed pump site was engine-validated.
        w,h=self.size(name,direction)
        for tree in self.ob.get('trees',[]):
            if abs(tree[1]-x)<w/2+.55 and abs(tree[2]-y)<h/2+.55:
                return tree[0]
        return None

    def reset_plan(self):
        self.stock=dict(self.ob.get('stock', {}))
        self.ops=[]
        self.transfer=100
        queue=self.ob.get('queue', [])
        self.worker_ticks=sum(int(q.get('ticks',0))*int(q.get('remaining',q.get('count',1))) for q in queue)
        self.worker_jobs=len(queue)
        self.worker_busy=bool(queue)
        self.parallel_worker=any(e.get('name')=='assembling-machine-1' for e in self.ob.get('entities',[]))
        self.pending={}
        for q in queue:
            recipe=self.recipes.get(q.get('recipe',''),{})
            for product in recipe.get('products',[]):
                item=product.get('name')
                self.pending[item]=self.pending.get(item,0)+product.get('amount',1)*q.get('remaining',q.get('count',1))
        self.wanted={}
        self.reserve={}
        # Protect the next construction's direct ingredients from production.
        for task in self.tasks[self.index:]:
            if task[0]!='place': continue
            if self.entity(*task[1:4]): continue
            if self.stock.get(task[1],0)>0: continue
            recipe=self.recipes.get(task[1],{})
            self.reserve={i['name']:i['amount'] for i in recipe.get('ingredients',[])}
            break

    def add(self, op):
        if len(self.ops)<31:
            self.ops.append(op)
            return True
        return False

    def move(self, e, role, item, count, extracting=False):
        count=max(0,min(int(count),self.transfer))
        if not extracting:
            count=min(count,int(self.stock.get(item,0)))
        if count and self.add(f"{'extract' if extracting else 'insert'} {e['id']} {role} {item} {count}"):
            self.stock[item]=self.stock.get(item,0)+(count if extracting else -count)
            self.transfer-=count
            return count
        return 0

    def gather(self, item, n=15):
        if self.worker_busy:
            return False
        if item=='wood':
            trees=self.ob.get('trees', [])
            if trees:
                target=trees[0][0]
                for t in self.tasks[self.index:]:
                    if t[0]=='place':
                        blocking=self.blocking_tree(*t[1:])
                        if blocking is not None:
                            target=blocking
                            break
                self.add(f'mine_tree {target}')
                self.worker_busy=True
                self.parallel_worker=False
                return True
        rows=[r for r in self.ob.get('resources', []) if r[1]==item and r[4]>0]
        if rows:
            r=max(rows,key=lambda r:r[4])
            self.add(f'mine_resource {r[0]} {max(1,min(15,int(n),r[4]))}')
            self.worker_busy=True
            self.parallel_worker=False
            return True
        return False

    def ensure(self, item, count, depth=0):
        """Compile a paid dependency frontier, never using pending output."""
        if self.stock.get(item,0)>=count:
            return True
        self.wanted[item]=max(count,self.wanted.get(item,0))
        if self.stock.get(item,0)+self.pending.get(item,0)>=count:
            return False
        if depth>12 or (self.worker_busy and not self.parallel_worker) or self.worker_jobs>=28:
            return False
        if item in ('coal','stone','iron-ore','copper-ore','wood'):
            self.gather(item,max(10,count-self.stock.get(item,0)))
            return False
        recipe=self.recipes.get(item)
        if not recipe or recipe.get('category')!='crafting':
            return False
        if any(i.get('type','item')!='item' for i in recipe.get('ingredients', [])):
            return False
        out=sum(p.get('amount',1) for p in recipe.get('products',[]) if p.get('name')==item)
        ticks=max(1,math.ceil(float(recipe.get('seconds',.5))*60))
        available_ticks=1740-self.worker_ticks
        if available_ticks<ticks: return False
        n=min(math.ceil((count-self.stock.get(item,0))/max(1,out)),available_ticks//ticks)
        affordable=min((int(self.stock.get(i['name'],0)//i['amount']) for i in recipe.get('ingredients',[]) if i['amount']>0),default=n)
        if self.parallel_worker and affordable>0: n=min(n,affordable)
        for ing in recipe.get('ingredients',[]):
            if self.stock.get(ing['name'],0)<ing['amount']*n:
                self.ensure(ing['name'],ing['amount']*n,depth+1)
                return False
        batch={'pipe':16,'iron-gear-wheel':18,'copper-cable':24,'electronic-circuit':10,
               'stone-furnace':2,'small-electric-pole':3,'transport-belt':2}.get(item,n)
        n=max(n,min(batch,affordable,available_ticks//ticks))
        if self.add(f'handcraft {recipe["name"]} {n}'):
            for ing in recipe.get('ingredients',[]):
                self.stock[ing['name']]=self.stock.get(ing['name'],0)-ing['amount']*n
            self.worker_busy=True
            self.worker_ticks+=ticks*n
            self.worker_jobs+=1
            self.pending[item]=self.pending.get(item,0)+int(out)*n
        return False

    def craft_production(self, item, n):
        recipe=self.recipes.get(item)
        if not recipe or recipe.get('category')!='crafting' or n<=0: return
        ticks=max(1,math.ceil(recipe.get('seconds',.5)*60))
        n=min(n,(1740-self.worker_ticks)//ticks)
        for ing in recipe.get('ingredients',[]):
            n=min(n,max(0,int((self.stock.get(ing['name'],0)-self.reserve.get(ing['name'],0))//ing['amount'])))
        if n<=0 or self.worker_jobs>=28: return
        if self.add(f'handcraft {item} {n}'):
            for ing in recipe['ingredients']:
                self.stock[ing['name']]-=ing['amount']*n
            for product in recipe.get('products',[]):
                key=product['name']
                self.pending[key]=self.pending.get(key,0)+product.get('amount',1)*n
            self.worker_ticks+=ticks*n
            self.worker_jobs+=1
            self.worker_busy=True

    def service(self):
        entities=self.ob.get('entities', [])
        # Science has first claim on transfers. Material stockpiles are bounded.
        for e in entities:
            if e.get('name')=='assembling-machine-1':
                for item,n in inventory(e,'output').items(): self.move(e,'output',item,n,True)
        for e in entities:
            if e.get('name')=='wooden-chest' and self.stock.get('coal',0)<65:
                self.move(e,'chest','coal',min(inventory(e,'chest').get('coal',0),65-self.stock.get('coal',0)),True)
        for e in entities:
            if e.get('name') in ('burner-mining-drill','stone-furnace','boiler'):
                fuel=sum(inventory(e,'fuel').values())
                target=8 if e.get('name')!='boiler' else 12
                if fuel<2: self.move(e,'fuel','coal',target-fuel)
        labs=[e for e in entities if e.get('name')=='lab']
        current=self.ob.get('research',{}).get('current')
        details=next((r for r in self.ob.get('available_research',[]) if r.get('name')==current),{})
        seconds=max(1,float(details.get('seconds',30)))
        units=max(1,math.ceil(30/seconds))
        # Refill several turns at once, rather than one operation per missing
        # pack per lab per turn. Least-stocked laboratories receive priority.
        for pack in ('logistic-science-pack','automation-science-pack'):
            for e in sorted(labs,key=lambda e:inventory(e,'lab').get(pack,0)):
                count=inventory(e,'lab').get(pack,0)
                if count<units+1:
                    self.move(e,'lab',pack,3*units+1-count)
        self.feed_machines()
        for e in entities:
            if e.get('name')=='stone-furnace':
                for item,n in inventory(e,'result').items():
                    limit=180 if item=='iron-plate' else 120
                    self.move(e,'result',item,min(n,max(0,limit-self.stock.get(item,0))),True)

    def research(self):
        if self.ob.get('research',{}).get('current'): return
        green=self.stock.get('logistic-science-pack',0)+sum(inventory(e,'lab').get('logistic-science-pack',0) for e in self.ob.get('entities',[]) if e.get('name')=='lab')
        # Do not commit to a long red-only project just as the green factory
        # is commissioning. A bounded grace period lets the first packs arrive.
        if green:
            self.green_wait=0
        elif 'logistic-science-pack' in self.recipes and self.green_wait<8:
            self.green_wait+=1
            return
        choices=[]
        for r in self.ob.get('available_research',[]):
            ing=r.get('ingredients',[])
            if isinstance(ing,dict): ing=[{'name':k,'amount':v} for k,v in ing.items()]
            names={i.get('name') for i in ing}
            allowed={'automation-science-pack','logistic-science-pack'} if green else {'automation-science-pack'}
            if not names or not names<=allowed or r.get('researched'): continue
            priorities={'automation':0,'logistics':1,'logistic-science-pack':2}
            value=sum((2 if i['name']=='logistic-science-pack' else 1)*i.get('amount',1) for i in ing)
            choices.append((priorities.get(r['name'],3),-value/max(1,r.get('seconds',30)),r.get('units',100),r['name']))
        if choices: self.add('set_research '+min(choices)[3])

    def task(self):
        cursor=self.index
        while cursor<len(self.tasks) and len(self.ops)<28:
            t=self.tasks[cursor]
            if t[0]=='place':
                _,name,x,y,d=t
                if self.entity(name,x,y):
                    if cursor==self.index:
                        self.index+=1
                    cursor+=1
                    continue
                tree=self.blocking_tree(name,x,y,d)
                if tree is not None:
                    if not self.worker_busy:
                        self.add(f'mine_tree {tree}')
                        self.worker_busy=True
                        self.parallel_worker=False
                    return
                if self.ensure(name,1):
                    self.add(f'place {name} {x:g} {y:g} {d}')
                    self.stock[name]-=1
                    cursor+=1
                    continue
                return
            # Barriers require newly placed machines to be observed first.
            if cursor!=self.index:
                return
            if t[0]=='science':
                completed=self.ob.get('research',{}).get('completed',{})
                if 'automation' in completed or self.stock.get('automation-science-pack',0)+sum(inventory(e,'lab').get('automation-science-pack',0) for e in self.ob.get('entities',[]) if e.get('name')=='lab')>=t[1]:
                    self.index+=1
                    cursor+=1
                    continue
                self.ensure('automation-science-pack',t[1])
                return
            if t[0]=='research':
                research=self.ob.get('research',{})
                if t[1] not in research.get('completed',[]) and research.get('current')!=t[1]:
                    self.add('set_research '+t[1])
                self.index+=1
                cursor+=1
                continue
            if t[0]=='unlock':
                if t[1] in self.recipes:
                    self.index+=1
                    cursor+=1
                    continue
                return

    def feed_machines(self):
        machines=[e for e in self.ob.get('entities',[]) if e.get('name')=='assembling-machine-1']
        for e in machines:
            recipe=self.machine_recipes.get(pos(e),'automation-science-pack')
            if recipe not in self.recipes: continue
            if e.get('recipe')!=recipe:
                if not e.get('recipe'): self.add(f"set_recipe {e['id']} {recipe}")
                else: continue
            for ing in self.recipes[recipe].get('ingredients',[]):
                item=ing['name']
                n=min(4*int(ing['amount'])-inventory(e,'input').get(item,0),max(0,self.stock.get(item,0)-self.reserve.get(item,0)))
                self.move(e,'input',item,n)

    def produce(self):
        machines=[e for e in self.ob.get('entities',[]) if e.get('name')=='assembling-machine-1']
        green_machines=[e for e in machines if self.machine_recipes.get(pos(e))=='logistic-science-pack' and 'logistic-science-pack' in self.recipes]
        if not machines: return
        if not self.parallel_worker: return
        # Maintain several turns of independent intermediates; pending products
        # count toward a target but are never spendable by another craft.
        targets=[('iron-gear-wheel',30),('copper-cable',42),('electronic-circuit',12),('inserter',12),('transport-belt',12)] if green_machines else [('iron-gear-wheel',18)]
        for item,target in targets:
            recipe=self.recipes.get(item)
            if not recipe: continue
            have=self.stock.get(item,0)+self.pending.get(item,0)
            need=max(0,target+self.reserve.get(item,0)-have)
            out=sum(p.get('amount',1) for p in recipe.get('products',[]) if p.get('name')==item)
            if need: self.craft_production(item,math.ceil(need/max(1,out)))
        red=self.stock.get('automation-science-pack',0)+sum(inventory(e,'lab').get('automation-science-pack',0) for e in self.ob.get('entities',[]) if e.get('name')=='lab')
        green=self.stock.get('logistic-science-pack',0)+sum(inventory(e,'lab').get('logistic-science-pack',0) for e in self.ob.get('entities',[]) if e.get('name')=='lab')
        # Extra red is useful while lab demand remains; never spend scarce gear
        # reserves that keep both green input factories and construction alive.
        if red<max(12,green+6):
            old=self.reserve.get('iron-gear-wheel',0)
            self.reserve['iron-gear-wheel']=max(old,8 if green_machines else 3)
            self.craft_production('automation-science-pack',3)
            self.reserve['iron-gear-wheel']=old

    def compact(self):
        keep=('turn','remaining_turns','stock','queue','entities','research','available_research','production','lab_consumption','score','results','pump_site')
        result={k:self.ob.get(k) for k in keep}
        result['goal_index']=self.index
        result['goals_remaining']=self.tasks[self.index:self.index+5]
        result['wanted']=self.wanted
        result['recent_failures']=self.failures[-6:]
        result['enabled_recipes']=[r for r in self.recipes.values() if r['name'] in ('lab','boiler','steam-engine','offshore-pump','small-electric-pole','automation-science-pack','assembling-machine-1')]
        result['patches']=self.ob.get('patches')
        result['mining_ids']={k:v[0] for k,v in self.sites.items() if v}
        return result

    def advise(self):
        remaining=self.ctx.budget.remaining()
        if self.calls>=5 or remaining.get('system2_calls',0)<=1 or remaining.get('wall_s',0)<240:
            return
        self.calls+=1
        self.last_call=self.ob.get('turn',0)
        prompt='''You are the exception planner for a deterministic Factorio factory controller.
The executor builds burner drill -> furnace pairs, remotely refuels/extracts with a 100-item shared quota, compiles exact recipes, then red and green assemblers with a manual red supplement and five labs. Normal operations are automatic. Diagnose only the specific blockage shown. Think about physical fluid connections and electricity if power is idle. 1800 ticks/turn; no avatar, no distance limit. Stable entity IDs are in the observation. The grammar accepts full action strings, not code. Return JSON {"diagnosis":"brief", "actions":["one full action batch", ...]} with at most 6 turns of corrective actions, or an empty list if healthy. Preserve buildings unless you have a concrete repair. Recover before replacing; mined buildings must be empty. You may use mine_resource ID COUNT, mine_tree ID, handcraft RECIPE COUNT, place ITEM X Y DIR, rotate ID, set_recipe ID RECIPE, insert/extract ID ROLE ITEM COUNT, mine_building ID, set_research NAME, wait. Counts are integers; names use hyphens. Queries must stand alone. Don't invent new IDs. Craft intermediates in separate turns. No raw prose outside JSON.
'''+json.dumps(self.compact(),separators=(',',':'))
        self.advisor_goal = self.index
        def call():
            try:
                result=self.ctx.think(prompt,timeout_s=min(180,remaining.get('wall_s',300)-40))
                text=result.text
                start=text.find('{');end=text.rfind('}')
                return json.loads(text[start:end+1])
            except Exception as exc:
                self.ctx.log('advisor_failed',error=type(exc).__name__)
                return {}
        self.advisor_future=self.ctx.spawn(call)

    def collect_advice(self):
        if self.advisor_future is None or not self.advisor_future.done():
            return
        try:
            data=self.advisor_future.result()
            if isinstance(data,dict) and self.index==self.advisor_goal and self.ob.get('turn',0)-self.last_call<=16:
                self.repairs=self.parse(data.get('actions',[]))[:6]
                self.ctx.log('advisor',diagnosis=str(data.get('diagnosis',''))[:500],actions=self.repairs)
            else:
                self.ctx.log('advisor_discarded',reason='state advanced')
        except Exception as exc:
            self.ctx.log('advisor_failed',error=type(exc).__name__)
        self.advisor_future=None

    @staticmethod
    def parse(actions):
        if isinstance(actions,str):
            actions=[actions]
        if not isinstance(actions,list):
            return []
        name=r'[a-z][a-z0-9-]*';num=r'-?\d+(?:\.\d+)?';ident=r'\d+'
        patterns=[r'wait',fr'mine_resource {ident} {ident}',fr'mine_tree {ident}',fr'handcraft {name} {ident}',fr'place {name} {num} {num}(?: (?:0|4|8|12))?',fr'rotate {ident}',fr'set_recipe {ident} {name}',fr'(?:insert|extract) {ident} (?:fuel|source|result|input|output|lab|chest) {name} {ident}',fr'mine_building {ident}',fr'set_research {name}',fr'inspect {ident}',fr'(?:recipe|technology) {name}']
        out=[]
        for a in actions:
            if not isinstance(a,str):
                continue
            parts=[' '.join(p.split()) for p in a.strip().split(';') if p.strip()]
            if 0<len(parts)<=32 and all(any(re.fullmatch(pat,p) for pat in patterns) for p in parts):
                if len(parts)>1 and any(p.startswith(('inspect ','recipe ','technology ')) for p in parts):
                    continue
                out.append('; '.join(parts))
        return out

    def run_episode(self, game):
        self.ob=obj(self.ctx.observe())
        if not self.ob:
            return
        self.init_tasks()
        for _ in range(180):
            if self.ctx.done or self.ob.get('remaining_turns',1)<=0:
                return
            for r in self.ob.get('enabled_recipes',[]):
                if isinstance(r,dict):
                    self.recipes[r['name']]=r
            self.reset_plan()
            turn=self.ob.get('turn',0)
            errors=[r for r in self.ob.get('results',[]) if r.get('ok') is False]
            if errors:
                self.failures.append({'turn':turn,'action':self.last_action,'errors':errors})
            if self.index==self.last_index:
                self.blocked+=1
            else:
                self.blocked=0
            self.last_index=self.index
            # Model calls are reserved for unexpected evidence, never the happy path.
            self.collect_advice()
            if self.advisor_future is None and (self.blocked>=12 and self.index<len(self.tasks) or len(errors)>0) and turn-self.last_call>=16:
                self.advise()
                if self.ctx.done:
                    return
            if self.repairs:
                action=self.repairs.pop(0)
            else:
                self.service()
                self.research()
                # Keep fuel alive before investing in new machines.
                if self.stock.get('coal',0)<8:
                    self.gather('coal',15)
                self.task()
                self.produce()
                # Use otherwise-idle workforce to create intermediates for science.
                if not self.worker_busy and self.index>=len(self.tasks)-8 and self.stock.get('iron-gear-wheel',0)<12:
                    self.ensure('iron-gear-wheel',18)
                action='; '.join(self.ops) or 'wait'
            self.ctx.log('controller_state', turn=turn, goal=self.index, tasks=len(self.tasks), stock=self.stock, entities=self.ob.get('entities',[]), research=self.ob.get('research',{}), wanted=self.wanted, production=self.ob.get('production'), lab_consumption=self.ob.get('lab_consumption'), available_research=self.ob.get('available_research'), enabled_recipes=list(self.recipes.values()), results=self.ob.get('results'), worker_ticks=self.worker_ticks, transfer_left=self.transfer, reserve=self.reserve)
            self.last_action=action
            try:
                new=self.ctx.act(action)
                self.ob=obj(new if new is not None else self.ctx.observe())
            except Exception as exc:
                if self.ctx.done:
                    return
                self.ctx.log('action_exception',error=type(exc).__name__,action=action)
                try:
                    self.ob=obj(self.ctx.act('wait'))
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
