"""Demand-driven factory steward: science before surplus under a transfer quota."""
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
        # Recovery state is deliberately small and episode-local.  A failed
        # placement is evidence about this map, so remember alternatives rather
        # than repeatedly paying turns for the same coordinates.
        self.task_attempts = {}
        self.last_score = None
        self.last_progress = {}
        # Persistent engine statuses are a second failure channel: placement
        # may be accepted while a machine is outside the power/fluid graph.
        # Keep only compact episode-local evidence so ordinary input starvation
        # does not spend an Astra call.
        self.status_history = {}
        self.production_history = {}

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
        current=research.get('current')
        # The adaptor has emitted both a string and an opaque sentinel in
        # different observation revisions.  Treat only a named, non-complete
        # technology as active; a truthy opaque value must not freeze science.
        current_name = current if isinstance(current,str) else (
            current.get('name') if isinstance(current,dict) else None)
        completed=research.get('completed',{})
        if isinstance(completed,dict):
            completed_names={k for k,v in completed.items() if v}
        elif isinstance(completed,list):
            completed_names={x for x in completed if isinstance(x,str)}
        else:
            completed_names=set()
        if current_name and current_name not in ('none','null') and current_name not in completed_names:
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
            rname=r.get('name')
            if names and names <= allowed and not r.get('researched') and rname not in completed_names:
                # Unlock green immediately, then prefer research with the best
                # weighted consumption per lab second. Actual recipe tables
                # and prerequisite availability decide what is possible.
                priority={'automation':0,'logistics':1,'logistic-science-pack':2}
                value=sum((2 if i.get('name')=='logistic-science-pack' else 1)*i.get('amount',1) for i in ing)
                rate=value/max(1,float(r.get('seconds',30)))
                choices.append((priority.get(rname,3),-rate,r.get('units',100),rname))
        if choices:
            self.add('set_research '+min(choices)[3])

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
        keep=('turn','remaining_turns','stock','queue','entities','research','available_research','production','lab_consumption','score','results','pump_site')
        result={k:self.ob.get(k) for k in keep}
        result['goal_index']=self.index
        result['goals_remaining']=self.tasks[self.index:self.index+5]
        result['wanted']=self.wanted
        result['recent_failures']=self.failures[-6:]
        result['enabled_recipes']=[r for r in self.recipes.values() if r['name'] in ('lab','boiler','steam-engine','offshore-pump','small-electric-pole','automation-science-pack','assembling-machine-1','inserter','transport-belt','logistic-science-pack')]
        result['patches']=self.ob.get('patches')
        result['mining_ids']={k:v[0] for k,v in self.sites.items() if v}
        result['progress']=self.last_progress
        result['last_action']=self.last_action
        result['silent_faults']=self.silent_faults()
        return result

    def silent_faults(self):
        """Return repeated, actionable machine faults absent rejected ops.

        Input and science statuses are normal while the compiler transfers
        cargo. Escalate only repeated fuel/power faults, or a missing network
        id on a commissioned electric machine. Network id zero/None is
        evidence of a disconnected placement even when the action succeeded.
        """
        faults=[]
        entities=self.ob.get('entities', [])
        for e in entities:
            if not isinstance(e,dict):
                continue
            name=e.get('name'); eid=e.get('id')
            if name not in ('assembling-machine-1','lab','boiler','steam-engine') or eid is None:
                continue
            status=str(e.get('status_name',''))
            key=str(eid)
            sig=(status, e.get('electric_network_id'),
                 tuple(sorted(inventory(e,'input').items())),
                 tuple(sorted(inventory(e,'lab').items())))
            hist=self.status_history.setdefault(key,[])
            hist.append(sig)
            if len(hist)>6: del hist[:-6]
            # Electric consumers should share the generator network once the
            # power block exists.  Treat an absent/zero id as actionable only
            # after the first few turns, when bootstrap placement is complete.
            if name in ('assembling-machine-1','lab') and self.ob.get('turn',0)>=20:
                net=e.get('electric_network_id')
                if net in (None,0,'0') and len(hist)>=3:
                    faults.append({'id':eid,'name':name,'status':'disconnected_power',
                                   'network':net,'history':hist[-3:]})
                    continue
            if len(hist)<3 or len({x[0] for x in hist[-3:]})!=1:
                continue
            if status in ('no_power','no_fuel','missing_fuel'):
                faults.append({'id':eid,'name':name,'status':status,
                               'network':e.get('electric_network_id')})
            # Input and science-pack shortages are deliberately left to the
            # normal transfer compiler; they are transient while a batch is
            # being assembled and are too common to justify a slow call.
        return faults[:6]

    def local_repair(self, errors):
        """Turn concrete placement failures into a bounded map-local search.

        The normal compiler is intentionally optimistic because the scenario
        generator usually gives open corridors.  On a different coast, a
        tree, water edge, or footprint can invalidate one operation.  Trying a
        few nearby half-tile locations is cheaper and safer than asking Astra
        to replay the entire build order.
        """
        if not errors or not self.last_action:
            return
        ops=[p.strip() for p in self.last_action.split(';')]
        failed=[]
        for i,e in enumerate(errors):
            if isinstance(e,dict) and e.get('ok') is False:
                failed.append(i)
        for i in failed:
            if i>=len(ops) or not ops[i].startswith('place '):
                continue
            bits=ops[i].split()
            if len(bits)<4:
                continue
            name=bits[1]
            try:
                x=float(bits[2]); y=float(bits[3]); d=int(bits[4]) if len(bits)>4 else 0
            except (TypeError,ValueError):
                continue
            # Pump placement is engine-validated by the observation and should
            # never be moved blindly.  For all other buildings search around
            # the requested footprint, preserving orientation.
            if name=='offshore-pump':
                continue
            key=(name,round(x,2),round(y,2),d)
            n=self.task_attempts.get(key,0)
            offsets=[(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2),
                     (1,1),(-1,1),(1,-1),(-1,-1)]
            if n>=len(offsets):
                continue
            dx,dy=offsets[n]
            self.task_attempts[key]=n+1
            for j,t in enumerate(self.tasks):
                if t[0]=='place' and t[1]==name and abs(float(t[2])-x)<.1 and abs(float(t[3])-y)<.1:
                    self.tasks[j]=('place',name,x+dx,y+dy,d)
                    if j<self.index:
                        self.index=j
                    self.ctx.log('local_repair',item=name,from_xy=[x,y],to_xy=[x+dx,y+dy],error=str(errors[i])[:300])
                    break

    def advise(self):
        remaining=self.ctx.budget.remaining()
        if self.calls>=4 or remaining.get('system2_calls',0)<=1 or remaining.get('wall_s',0)<300:
            return
        self.calls+=1
        self.last_call=self.ob.get('turn',0)
        prompt='''You are the exception planner for a deterministic Factorio factory controller.
The executor builds burner drill -> furnace pairs, remotely refuels/extracts with a 100-item shared quota, compiles exact recipes, then powered red and logistics-science blocks. Normal operations are automatic. Diagnose only the specific blockage shown. Think about physical fluid connections and electricity if power is idle. 1800 ticks/turn; no avatar, no distance limit. Stable entity IDs are in the observation. The grammar accepts full action strings, not code. Return JSON {"diagnosis":"brief", "actions":["one full action batch", ...]} with at most 6 turns of corrective actions, or an empty list if healthy. Preserve buildings unless you have a concrete repair. Recover before replacing; mined buildings must be empty. You may use mine_resource ID COUNT, mine_tree ID, handcraft RECIPE COUNT, place ITEM X Y DIR, rotate ID, set_recipe ID RECIPE, insert/extract ID ROLE ITEM COUNT, mine_building ID, set_research NAME, wait. Counts are integers; names use hyphens. Queries must stand alone. Don't invent new IDs. Craft intermediates in separate turns. No raw prose outside JSON.
The evidence may include a silent_faults list: these are repeated engine statuses after accepted placement, so verify network/power/fluid connections and repair the named machine or its nearby support. Do not rebuild a healthy chain merely because one lab is temporarily missing a pack.
'''+json.dumps(self.compact(),separators=(',',':'))
        self.advisor_goal = self.index
        def call():
            try:
                # Keep failure recovery from consuming the entire episode.
                result=self.ctx.think(prompt,timeout_s=min(110,max(30,remaining.get('wall_s',300)-180)))
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
                self.local_repair(errors)
            try:
                self.last_progress=self.ctx.progress()
            except Exception:
                self.last_progress={}
            faults=self.silent_faults()
            if self.index==self.last_index:
                self.blocked+=1
            else:
                self.blocked=0
            self.last_index=self.index
            # Model calls are reserved for unexpected evidence, never the happy path.
            self.collect_advice()
            stagnant = (self.last_progress.get('since_obs_change',0)>=3 or
                        self.last_progress.get('since_score_change',0)>=8)
            if self.advisor_future is None and (len(errors)>0 or faults or (stagnant and self.index<len(self.tasks))) and turn-self.last_call>=4:
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
                if not self.worker_busy and not self.parallel_worker and self.index>=len(self.tasks)-8 and self.stock.get('iron-gear-wheel',0)<12:
                    self.ensure('iron-gear-wheel',18)
                action='; '.join(self.ops) or 'wait'
            self.ctx.log('controller_state', turn=turn, goal=self.index, tasks=len(self.tasks), stock=self.stock, entities=self.ob.get('entities',[]), research=self.ob.get('research',{}), wanted=self.wanted, production=self.ob.get('production'), lab_consumption=self.ob.get('lab_consumption'), available_research=self.ob.get('available_research'), enabled_recipes=list(self.recipes.values()), results=self.ob.get('results'))
            self.last_action=action
            self.last_score=self.ob.get('score')
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
