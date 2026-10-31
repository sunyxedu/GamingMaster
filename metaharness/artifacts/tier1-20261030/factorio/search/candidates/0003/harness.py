"""Verified factory compiler with paused, persistent exception replanning."""
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
        self.last_parts = []
        self.repair_pending_evidence = None
        self.last_illegal_actions = 0
        self.history = []
        self.advice_history = []
        self.last_wanted = {}
        self.action_exception = None
        self.last_was_query = False
        self.query_replan = False

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
        put('small-electric-pole',8,13)
        put('assembling-machine-1',8,16)
        put('small-electric-pole',3,13)
        put('lab',3,16)
        put('small-electric-pole',3,19)
        put('lab',3,22)
        put('small-electric-pole',8,19)
        # Commission green while the bootstrap still has spare plates;
        # replenishment capacity can be built concurrently with science.
        self.green_roles = {}
        put('assembling-machine-1',8,22)
        a,b=transform(8,22)
        self.green_roles[(round(a,3),round(b,3))]='logistic-science-pack'
        for _ in range(2):
            pair('iron-ore')
        # Three coal pairs cover the additional smelter and science load,
        # instead of repeatedly stealing the serial worker for emergency coal.
        for _ in range(2):
            x,y=site('coal')
            self.tasks.extend([('place','burner-mining-drill',x,y,4),
                               ('place','wooden-chest',x+1.5,y-.5,0)])
            self.chest_items[(x+1.5,y-.5)]='coal'
        put('small-electric-pole',8,25)
        put('assembling-machine-1',8,28)
        a,b=transform(8,28)
        self.green_roles[(round(a,3),round(b,3))]='logistic-science-pack'
        pair('copper-ore')
        put('small-electric-pole',3,25)
        put('lab',3,28)
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
        self.lab_added={}
        self.materials_collected=False
        self.worker_busy=bool(self.ob.get('queue'))
        self.wanted={}
        self.pending={}
        self.worker_ticks=0
        self.worker_jobs=0
        self.parallel_worker=any(e.get('name')=='assembling-machine-1'
            for e in self.ob.get('entities',[])) and not self.ob.get('queue')

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
                self.parallel_worker=False
                self.worker_busy=True
                return True
        rows=[r for r in self.ob.get('resources', []) if r[1]==item and r[4]>0]
        if rows:
            r=max(rows,key=lambda r:r[4])
            self.add(f'mine_resource {r[0]} {max(1,min(15,int(n),r[4]))}')
            self.parallel_worker=False
            self.worker_busy=True
            return True
        return False

    def ensure(self, item, count, depth=0):
        """Compile a stock target to its next feasible paid leaf job."""
        if self.stock.get(item,0)>=count:
            return True
        self.wanted[item]=max(count,self.wanted.get(item,0))
        if self.stock.get(item,0)+self.pending.get(item,0)>=count:
            return False
        if depth>12 or (self.worker_busy and not self.parallel_worker) or self.worker_jobs>=28:
            return False
        missing=count-self.stock.get(item,0)
        recipe=self.recipes.get(item)
        if item in ('iron-plate','copper-plate','stone-brick'):
            return False
        if item in ('coal','stone','iron-ore','copper-ore','wood'):
            self.gather(item,max(missing,10))
            return False
        if not recipe or recipe.get('category') not in ('crafting','advanced-crafting','crafting-with-fluid'):
            return False
        if any(i.get('type','item')!='item' for i in recipe.get('ingredients', [])):
            return False
        products=recipe.get('products', [])
        out=sum(p.get('amount',1) for p in products if p.get('name')==item)
        n=max(1,math.ceil(missing/max(1,out)))
        seconds=float(recipe.get('seconds',.5))
        time_left=(1740-self.worker_ticks)/60 if self.parallel_worker else 29
        if time_left<seconds:
            return False
        n=min(n,max(1,int(time_left/max(.1,seconds))))
        affordable=min((int(self.stock.get(i['name'],0)//i['amount']) for i in recipe.get('ingredients',[]) if i['amount']>0),default=n)
        if self.parallel_worker and affordable>0:
            n=min(n,affordable)
        for ing in recipe.get('ingredients', []):
            if self.stock.get(ing['name'],0)<ing['amount']*n:
                self.ensure(ing['name'],ing['amount']*n,depth+1)
                return False
        # Batch only already-affordable common components; never wait for surplus.
        batch={'pipe':16,'iron-gear-wheel':18,'copper-cable':30,'electronic-circuit':10,
               'stone-furnace':2,'small-electric-pole':3,'transport-belt':2}.get(item,n)
        affordable=min((int(self.stock.get(i['name'],0)//i['amount']) for i in recipe.get('ingredients',[]) if i['amount']>0),default=n)
        n=max(n,min(batch,affordable,max(1,int(time_left/max(.1,seconds)))))
        if self.add(f'handcraft {recipe["name"]} {n}'):
            for ing in recipe.get('ingredients', []):
                self.stock[ing['name']]=self.stock.get(ing['name'],0)-ing['amount']*n
            self.worker_busy=True
            self.worker_ticks+=int(math.ceil(seconds*60))*n
            self.worker_jobs+=1
            self.pending[item]=self.pending.get(item,0)+int(out)*n
        return False

    def service(self):
        """Spend scarce transfers on useful outputs and consumption first.

        Furnace output is a free buffer. Raw material extraction is delayed
        until science has its inputs; large stockpiles are not a goal.
        """
        entities=self.ob.get('entities', [])
        # Unblock scored machines before collecting bulk plates.
        for e in entities:
            if e.get('name')=='assembling-machine-1':
                for item,n in inventory(e,'output').items():
                    self.move(e,'output',item,n,True)
        # Gather enough restart fuel, without filling a large coal stock every turn.
        fuel_need=sum(max(0,(8 if e['name']!='boiler' else 12)-sum(inventory(e,'fuel').values()))
                      for e in entities if e.get('name') in ('burner-mining-drill','stone-furnace','boiler')
                      and sum(inventory(e,'fuel').values())<2)
        coal_target=max(60,fuel_need+16)
        for e in entities:
            if e.get('name')=='wooden-chest':
                n=inventory(e,'chest').get('coal',0)
                self.move(e,'chest','coal',min(n,coal_target-self.stock.get('coal',0)),True)
        for e in entities:
            if e.get('name') in ('burner-mining-drill','stone-furnace','boiler'):
                amount=sum(inventory(e,'fuel').values())
                if amount<2:
                    self.move(e,'fuel','coal',(12 if e['name']=='boiler' else 8)-amount)
        labs=[e for e in entities if e.get('name')=='lab']
        # Fill evenly in rounds: a scarce green pack should activate an empty
        # lab before being banked behind another lab's already-loaded pack.
        for pack in ('logistic-science-pack','automation-science-pack'):
            allocation={e['id']:0 for e in labs}
            available=min(self.transfer,int(self.stock.get(pack,0)))
            for level in (1,2,3):
                for e in labs:
                    present=inventory(e,'lab').get(pack,0)+allocation[e['id']]
                    if available>0 and present<level:
                        allocation[e['id']]+=1
                        available-=1
            for e in labs:
                self.move(e,'lab',pack,allocation[e['id']])
        # The bootstrap needs immediate plate extraction to compile its first
        # machines. After commissioning, science feeds reserve their quota first.
        if not any(e.get('name')=='assembling-machine-1' for e in entities):
            self.collect_materials()

    def collect_materials(self):
        if self.materials_collected:
            return
        self.materials_collected=True
        targets={'iron-plate':220 if self.index<len(self.tasks) else 100,
                 'copper-plate':180 if self.index<len(self.tasks) else 80}
        furnaces=[e for e in self.ob.get('entities',[]) if e.get('name')=='stone-furnace']
        # Prioritize the most depleted material relative to its desired buffer.
        furnaces.sort(key=lambda e:min((self.stock.get(i,0)/targets.get(i,50)
                                       for i in inventory(e,'result')),default=9))
        for e in furnaces:
            for item,n in inventory(e,'result').items():
                self.move(e,'result',item,min(n,targets.get(item,30)-self.stock.get(item,0)),True)
        for e in furnaces:
            ore=self.furnace_ores.get(pos(e))
            if ore and sum(inventory(e,'source').values())<3 and self.stock.get(ore,0):
                self.move(e,'source',ore,min(15,self.stock[ore]))

    def research(self):
        research=self.ob.get('research',{})
        if research.get('current'):
            return
        green_supply=self.stock.get('logistic-science-pack',0)+sum(
            inventory(e,'lab').get('logistic-science-pack',0)
            for e in self.ob.get('entities',[]) if e.get('name')=='lab')
        allowed={'automation-science-pack'}
        if green_supply>0:
            allowed.add('logistic-science-pack')
        choices=[]
        for r in self.ob.get('available_research',[]):
            ing=r.get('ingredients',[])
            if isinstance(ing,dict):
                ing=[{'name':k,'amount':v} for k,v in ing.items()]
            names={i.get('name') for i in ing if isinstance(i,dict)}
            if names and names <= allowed and not r.get('researched'):
                # Unlock green immediately, then prefer research with the best
                # weighted consumption per lab second. Actual recipe tables
                # and prerequisite availability decide what is possible.
                priority={'automation':0,'logistics':1,'logistic-science-pack':2}
                value=sum((2 if i.get('name')=='logistic-science-pack' else 1)*i.get('amount',1) for i in ing)
                rate=value/max(1,float(r.get('seconds',30)))
                choices.append((priority.get(r['name'],3),-rate,r.get('units',100),r['name']))
        if choices:
            self.add('set_research '+min(choices)[3])

    def task(self):
        cursor=self.index
        while cursor<len(self.tasks) and len(self.ops)<28:
            t=self.tasks[cursor]
            if t[0]=='skip':
                if cursor==self.index:
                    self.index+=1
                cursor+=1
                continue
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
                        self.parallel_worker=False
                        self.worker_busy=True
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

    def produce(self):
        """The worker supplies cheap intermediates; engine makes scored packs."""
        machines=[e for e in self.ob.get('entities',[]) if e.get('name')=='assembling-machine-1']
        green=[]
        for e in machines:
            x,y=pos(e)
            role=next((r for (gx,gy),r in self.green_roles.items()
                       if abs(x-gx)<.15 and abs(y-gy)<.15),None)
            recipe_name=role or 'automation-science-pack'
            if recipe_name not in self.recipes:
                # Never fill a future green machine with red cargo: recipe
                # changes require it to be empty and idle.
                continue
            if e.get('recipe')!=recipe_name:
                if not e.get('recipe'):
                    self.add(f"set_recipe {e['id']} {recipe_name}")
                else:
                    continue
            for ing in self.recipes[recipe_name].get('ingredients',[]):
                if ing.get('type','item')!='item': continue
                # A turn of coverage plus one craft, with red backpressure.
                backlog=self.stock.get('automation-science-pack',0)
                cover=6 if recipe_name=='logistic-science-pack' else (1 if backlog>30 else 6)
                target=int(ing.get('amount',1))*cover
                self.move(e,'input',ing['name'],target-inventory(e,'input').get(ing['name'],0))
            if role: green.append(e)
        self.collect_materials()
        # Intermediate preparation never takes priority over unfinished
        # construction or emergency fuel. Target batches amortize worker jobs.
        if machines and self.stock.get('iron-gear-wheel',0)<18:
            self.ensure('iron-gear-wheel',36 if green else 18)
        if green:
            for item,target in [('copper-cable',60),('electronic-circuit',16),
                                ('inserter',12),('transport-belt',12)]:
                self.ensure(item,target)
            # Engine packs score production plus consumption. Spare worker
            # time can also turn existing ingredients into extra consumption
            # credit, using red surplus that would otherwise wait in stock.
            green_waiting=self.stock.get('logistic-science-pack',0)+sum(
                inventory(e,'lab').get('logistic-science-pack',0)
                for e in self.ob.get('entities',[]) if e.get('name')=='lab')
            if len(green)<2 and self.stock.get('automation-science-pack',0)>=6 and green_waiting<8:
                self.ensure('logistic-science-pack',self.stock.get('logistic-science-pack',0)+2)
        if not machines and self.index>=len(self.tasks):
            self.ensure('automation-science-pack',10)

    def compact(self):
        keep=('turn','remaining_turns','stock','queue','entities','research','available_research','production','lab_consumption','score','results','pump_site','bounds','water_rows','trees','query')
        result={k:self.ob.get(k) for k in keep}
        result['goal_index']=self.index
        result['goals_remaining']=[{'index':i,'task':t} for i,t in enumerate(self.tasks) if i>=self.index]
        result['wanted']=self.last_wanted
        result['history']=self.history[-10:]
        result['previous_advice']=self.advice_history[-3:]
        result['last_action']=self.last_action
        result['last_operations']=[{'op':p,'result':r} for p,r in zip(self.last_parts,self.ob.get('results',[]))]
        result['layout_issues']=self.layout_issues()
        result['recent_failures']=self.failures[-6:]
        result['repair_evidence']=self.repair_pending_evidence
        result['enabled_recipes']=list(self.recipes.values())
        result['patches']=self.ob.get('patches')
        result['mining_ids']={k:next((r for r in self.ob.get('resources',[]) if r[1]==k and r[4]>0),None) for k in self.sites}
        return result

    def layout_issues(self):
        water={r['y']:r.get('runs',[]) for r in self.ob.get('water_rows',[])}
        bounds=self.ob.get('bounds',[-128,-128,128,128])
        if not isinstance(bounds,list) or len(bounds)!=4:
            bounds=[-128,-128,128,128]
        issues=[]
        for index,t in enumerate(self.tasks):
            if index<self.index or t[0]!='place' or t[1]=='offshore-pump':
                continue
            _,name,x,y,d=t
            w,h=self.size(name,d)
            tiles=[(a,b) for a in range(math.floor(x-w/2+.01),math.floor(x+w/2-.01)+1)
                         for b in range(math.floor(y-h/2+.01),math.floor(y+h/2-.01)+1)]
            bad=[(a,b) for a,b in tiles if a<bounds[0] or b<bounds[1] or a>=bounds[2] or b>=bounds[3]
                 or any(lo<=a<=hi for lo,hi in water.get(b,[]))]
            if bad:
                issues.append({'index':index,'task':t,'water_or_boundary_tiles':bad})
        return issues

    def health(self, progress):
        turn=self.ob.get('turn',0)
        packs=('automation-science-pack','logistic-science-pack')
        production=self.ob.get('production',{})
        consumed=self.ob.get('lab_consumption',{})
        snapshot={'turn':turn,'goal':self.index,'score':self.ob.get('score'),
                  'production':{k:production.get(k,0) for k in (*packs,'iron-plate','copper-plate')},
                  'consumption':{k:consumed.get(k,0) for k in packs},
                  'stock':{k:self.ob.get('stock',{}).get(k,0) for k in (*packs,'iron-plate','coal')},
                  'assemblers':[e['id'] for e in self.ob.get('entities',[]) if e.get('name')=='assembling-machine-1'],
                  'labs':[e['id'] for e in self.ob.get('entities',[]) if e.get('name')=='lab'],
                  'statuses':{str(e['id']):e.get('status_name',e.get('status')) for e in self.ob.get('entities',[])}}
        if not self.history or self.history[-1]['turn']!=turn:
            self.history.append(snapshot)
            self.history=self.history[-16:]
        reasons=[]
        if self.action_exception:
            reasons.append({'action_exception':self.action_exception})
        if self.query_replan:
            reasons.append('Requested read-only evidence is available; continue the repair.')
        issues=self.layout_issues()
        if issues:
            reasons.append({'unbuildable_pending_layout':issues})
        if self.blocked>=12 and self.index<len(self.tasks):
            reasons.append('Current construction milestone has not advanced in 12 turns. Check resource contention as well as geometry.')
        if progress.get('since_obs_change',0)>=4 or progress.get('since_score_change',0)>=12:
            reasons.append({'progress_stall':progress})
        if len(self.history)>=7:
            old=self.history[-7]
            if set(snapshot['assemblers']) & set(old.get('assemblers',[])):
                if all(production.get(k,0)==old['production'].get(k,0) for k in packs):
                    reasons.append('No engine science production across six turns despite installed assemblers.')
            labs=[e for e in self.ob.get('entities',[]) if e.get('name')=='lab']
            stocked=any(sum(inventory(e,'lab').values())>0 for e in labs)
            if stocked and set(snapshot['labs']) & set(old.get('labs',[])) and self.ob.get('research',{}).get('current') and all(consumed.get(k,0)==old['consumption'].get(k,0) for k in packs):
                reasons.append('Stocked labs consumed no science in six turns; check missing pack types, network and generation.')
        return reasons

    def valid_task(self, t):
        if not isinstance(t, (list, tuple)) or not t:
            return None
        if t[0] == 'place' and len(t) == 5:
            name, x, y, direction = t[1:]
            if (not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9-]*', name)
                or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                or not math.isfinite(x) or not math.isfinite(y)
                or not (-128 <= x < 128 and -128 <= y < 128)
                or direction not in (0, 4, 8, 12)):
                return None
            return tuple(t)
        if t[0] in ('research', 'unlock') and len(t) == 2 and isinstance(t[1], str):
            if re.fullmatch(r'[a-z][a-z0-9-]*', t[1]):
                return tuple(t)
        if t[0] == 'science' and len(t) == 2 and isinstance(t[1], int) and 1 <= t[1] <= 100:
            return tuple(t)
        if t == ['skip']:
            return ('skip',)
        return None

    def apply_edits(self, data):
        # Edits name the expected old tuple, making them compare-and-swap patches.
        # Role maps follow relocated buildings rather than the old coordinates.
        edits=data.get('task_edits', [])
        additions=data.get('insert_tasks', [])
        if not isinstance(edits,list) or len(edits)>16 or not isinstance(additions,list) or len(additions)>12:
            return False
        validated=[]
        seen=set()
        for edit in edits:
            if not isinstance(edit,dict):
                return False
            index=edit.get('index')
            if not isinstance(index,int) or not self.index <= index < len(self.tasks) or index in seen:
                return False
            seen.add(index)
            old=self.tasks[index]
            if edit.get('expected') != list(old):
                return False
            new=self.valid_task(edit.get('task'))
            if new is None:
                return False
            validated.append((index,old,new))
        valid=[self.valid_task(t) for t in additions]
        if any(t is None for t in valid):
            return False
        # Validate the whole patch before changing any task. Update role maps
        # from their original contents so a swap cannot overwrite another role.
        moves={}
        for index,old,new in validated:
            if old[0]=='place' and new[0]=='place' and old[1]==new[1]:
                oldpos=(old[2],old[3]); newpos=(new[2],new[3])
                moves[oldpos]=newpos
        for mapping in (self.furnace_ores,self.chest_items,self.green_roles):
            relocated={moves.get(k,k):v for k,v in mapping.items()}
            mapping.clear()
            mapping.update(relocated)
        for index,old,new in validated:
            self.tasks[index]=new
            self.ctx.log('task_edit',index=index,old=old,new=new)
        if valid:
            self.tasks[self.index:self.index]=valid
            self.ctx.log('insert_tasks',tasks=valid)
        return True

    def advise(self):
        remaining=self.ctx.budget.remaining()
        if self.calls>=6 or remaining.get('system2_calls',0)<=1 or remaining.get('wall_s',0)<330:
            return False
        self.calls+=1
        self.last_call=self.ob.get('turn',0)
        prompt='''You repair a turn-based Factorio factory compiler. The game is PAUSED during this
call. Do not return code. Return JSON only with:
{"diagnosis":"specific cause and expected observable recovery",
 "actions":["one action batch", ...],
 "task_edits":[{"index":42,"expected":["place","lab",1.5,2.5,0],
                "task":["place","lab",4.5,2.5,0]}],
 "insert_tasks":[["place","pipe",2.5,3.5,0]]}
All arrays may be empty. At most 4 action batches, 16 edits and 12 inserted tasks.
The current compiled plan builds fueled burner-drill/furnace pairs, coal chests,
pump/pipes/boiler/engine/poles/labs, then red/green science. Every turn it services
machines, selects research, compiles building recipes from observed ingredients,
and handcrafts science intermediates. It verifies each task by finding the item
at its exact target coordinate. Therefore a moved placement MUST also update its
task coordinate using task_edits. Edits can affect any pending task. Expected must
exactly equal the old task. insert_tasks run before the current task and automatically
craft missing buildings. Place tasks are ["place",item,x,y,direction]. Other tasks:
["science",red_count], ["research",technology], ["unlock",recipe], ["skip"]. Skip only
redundant or optional tasks with an explicit explanation. Move linked machines
TOGETHER and preserve ports, power coverage and drill output geometry. Already
placed buildings need actual recovery actions; editing tasks alone does not move them.
Current coordinates may collide with water, bounds, or trees. Terrain rows are inclusive.
Directions 0/4/8/12=N/E/S/W. Odd buildings use half-tile centers. Drill and furnace
are 2x2; lab/assembler 3x3; boiler 3x2 (rotates); engine 3x5 (rotates).
The pump_site is validated for the pump only, NOT all downstream factory buildings.
Only use observed IDs. A raw action batch is at most 32 operations; transfer quota
is 100 actual inserted+extracted items total. Crafts reserve existing inputs and
complete after stepping; no recursive crafting. Count means recipe executions.
One batch advances exactly 30 seconds regardless of how many wait operations it
contains. Separate turns must be separate entries in the actions array.
Allowed: wait; mine_resource ID COUNT; mine_tree ID; handcraft RECIPE COUNT;
place ITEM X Y DIR; rotate ID; set_recipe ID RECIPE;
insert/extract ID ROLE ITEM COUNT; mine_building ID; set_research NAME.
Roles fuel/source/result/input/output/lab/chest. Mine only EMPTY buildings; fluids
and burning fuel are lost. Read-only queries recipe NAME, technology NAME, inspect
OFFSET must be alone. These return evidence for a subsequent call. Recipe and
research names must be observed or queried, not assumed. Do not repeat failed
operations unchanged. A low iron stock with growing science score can be healthy;
a late extra lab can wait when green packs, not lab capacity, are limiting.
Prioritize concrete recovery over expansion. Explain how the repaired plan will
be recognized as working from the next observations.
'''+json.dumps(self.compact(),separators=(',',':'))
        try:
            response=self.ctx.think(prompt,timeout_s=min(300.0,remaining['wall_s']-30))
            raw=response.text
            start=raw.find('{'); end=raw.rfind('}')
            data=json.loads(raw[start:end+1])
            if not isinstance(data,dict):
                return False
            if not self.apply_edits(data):
                self.ctx.log('advisor_invalid_patch')
                return False
            self.advice_history.append({'turn':self.ob.get('turn'), 'diagnosis':str(data.get('diagnosis',''))[:1200]})
            self.repairs=self.parse(data.get('actions',[]))[:4]
            self.ctx.log('advisor',diagnosis=str(data.get('diagnosis',''))[:1200],actions=self.repairs)
            return bool(self.repairs or data.get('task_edits') or data.get('insert_tasks'))
        except Exception as exc:
            self.ctx.log('advisor_failed',error=type(exc).__name__)
            return False

    @staticmethod
    def parse(actions):
        if isinstance(actions,str):
            actions=[actions]
        if not isinstance(actions,list):
            return []
        name=r'[a-z][a-z0-9-]*';num=r'-?\d+(?:\.\d+)?';ident=r'\d+'
        patterns=[r'wait',fr'mine_resource {ident} {ident}',fr'mine_tree {ident}',fr'handcraft {name} {ident}',fr'place {name} {num} {num}(?: (?:0|4|8|12))?',fr'rotate {ident}',fr'set_recipe {ident} {name}',fr'(?:insert|extract) {ident} (?:fuel|source|result|input|output|lab|chest) {name} {ident}',fr'mine_building {ident}',fr'set_research {name}',fr'inspect {ident}',fr'(?:recipe|technology) {name}']
        out=[]
        for a in actions[:4]:
            if not isinstance(a,str):
                break
            parts=[' '.join(p.split()) for p in a.strip().split(';') if p.strip()]
            if not (0<len(parts)<=32 and all(any(re.fullmatch(pat,p) for pat in patterns) for p in parts)):
                break
            if len(parts)>1 and any(p.startswith(('inspect ','recipe ','technology ')) for p in parts):
                break
            if all(p=='wait' for p in parts):
                parts=['wait']
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
                self.repairs=[]  # A failed repair invalidates its dependent suffix.
            if self.index==self.last_index:
                self.blocked+=1
            else:
                self.blocked=0
            self.last_index=self.index
            # Model calls are reserved for unexpected evidence, never the happy path.
            progress = {}
            try:
                progress = self.ctx.progress() or {}
            except Exception:
                progress = {}
            illegal = int(progress.get('illegal_actions', 0) or 0)
            # A rejected placement is a transactional failure: replan on the
            # next boundary instead of replaying it for twelve turns.
            urgent = bool(errors) or illegal > self.last_illegal_actions
            self.last_illegal_actions=max(self.last_illegal_actions,illegal)
            reasons=self.health(progress)
            if urgent:
                reasons.append({'rejected_operations':errors})
            cooldown=0 if self.query_replan else (1 if urgent else 12)
            if reasons and not self.repairs and turn-self.last_call>=cooldown:
                self.repair_pending_evidence={'turn':turn,'goal':self.index,'action':self.last_action,
                                              'errors':errors,'progress':progress,'reasons':reasons}
                self.advise()
                self.query_replan=False
                self.action_exception=None
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
                if not self.worker_busy and not self.parallel_worker and self.index>=len(self.tasks)-8 and self.stock.get('iron-gear-wheel',0)<12:
                    self.ensure('iron-gear-wheel',18)
                action='; '.join(self.ops) or 'wait'
            self.ctx.log('controller_state', turn=turn, goal=self.index, tasks=len(self.tasks), stock=self.stock, entities=self.ob.get('entities',[]), research=self.ob.get('research',{}), wanted=self.wanted, production=self.ob.get('production'), lab_consumption=self.ob.get('lab_consumption'), available_research=self.ob.get('available_research'), enabled_recipes=list(self.recipes.values()), results=self.ob.get('results'))
            self.last_wanted=dict(self.wanted)
            self.last_action=action
            self.last_was_query=action.startswith(('recipe ','technology ','inspect '))
            self.last_parts=[p.strip() for p in action.split(';') if p.strip()]
            try:
                new=self.ctx.act(action)
                self.ob=obj(new if new is not None else self.ctx.observe())
                self.query_replan=self.last_was_query and not self.repairs
                self.ctx.log('step_verification',turn=self.ob.get('turn'),progress=self.ctx.progress(),
                             results=self.ob.get('results'),repair_remaining=len(self.repairs))
            except Exception as exc:
                if self.ctx.done:
                    return
                self.action_exception={'error':type(exc).__name__,'message':str(exc)[:300],'action':action}
                self.repairs=[]
                self.ctx.log('action_exception',error=type(exc).__name__,action=action)
                try:
                    self.ob=obj(self.ctx.act('wait'))
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
