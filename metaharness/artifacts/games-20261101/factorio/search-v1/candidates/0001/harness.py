"""Factorio command-preserving, receding-horizon factory supervisor."""
from __future__ import annotations
import json
import math
import re

NAME = r'[a-zA-Z0-9_-]+'
NUM = r'-?\d+(?:\.\d+)?'
REF = r'(?:\d+|@[a-zA-Z0-9_-]+:\d+)'
PATTERNS = {
 'wait': r'wait', 'mine_resource': rf'mine_resource {REF} [1-9]\d*',
 'mine_tree': rf'mine_tree {REF}', 'handcraft': rf'handcraft {NAME} [1-9]\d*',
 'place': rf'place {NAME} {NUM} {NUM}(?: (?:0|4|8|12))?',
 'rotate': rf'rotate {REF}', 'set_recipe': rf'set_recipe {REF} {NAME}',
 'insert': rf'insert {REF} (?:fuel|source|result|input|output|lab|chest) {NAME} [1-9]\d*',
 'extract': rf'extract {REF} (?:fuel|source|result|input|output|lab|chest) {NAME} [1-9]\d*',
 'mine_building': rf'mine_building {REF}', 'set_research': rf'set_research {NAME}',
 'inspect': r'inspect \d+', 'recipe': rf'recipe {NAME}', 'technology': rf'technology {NAME}',
 'service': r'service(?: [1-9]\d*)?'
}
READONLY = {'inspect','recipe','technology'}

def unpack(obs):
    if isinstance(obs, str):
        try: return json.loads(obs)
        except (ValueError, TypeError): return {}
    return obs if isinstance(obs, dict) else {}

def inv(e, role):
    d = e.get('inventories', e.get('inventory', {}))
    if not isinstance(d, dict): return {}
    v = d.get(role, {})
    if isinstance(v, dict): return {k:int(n) for k,n in v.items() if isinstance(n,(int,float))}
    if isinstance(v, list):
        return {x['name']:int(x.get('count',x.get('amount',0))) for x in v if isinstance(x,dict) and 'name' in x}
    return {}

def valid_line(line):
    if not isinstance(line,str) or len(line)>12000: return None
    line = line.strip().strip('`').strip()
    line = re.sub(r'^\s*(?:[-*]|\d+[.)])\s+', '', line)
    ops = []
    for op in line.split(';'):
        op = ' '.join(op.strip().split())
        if not op: continue
        verb = op.split()[0]
        if verb not in PATTERNS or not re.fullmatch(PATTERNS[verb], op): return None
        ops.append(op)
    if not ops or len(ops)>32: return None
    if any(x.split()[0] in READONLY or x.split()[0]=='service' for x in ops) and len(ops)!=1: return None
    return '; '.join(ops)

def parse_reply(text):
    obj = None
    decoder = json.JSONDecoder()
    for m in re.finditer(r'[\[{]', str(text)):
        try:
            v,_ = decoder.raw_decode(str(text)[m.start():])
            if isinstance(v,(dict,list)):
                obj=v; break
        except ValueError: pass
    if isinstance(obj,dict): raw=obj.get('actions',obj.get('plan',[]))
    elif isinstance(obj,list): raw=obj
    else: raw=str(text).splitlines()
    if isinstance(raw,str): raw=raw.splitlines()
    if not isinstance(raw,list): raw=[]
    actions=[]
    for x in raw[:24]:
        line=valid_line(x)
        if line:
            if re.fullmatch(r'service \d+',line):
                actions.extend(['service']*min(16,int(line.split()[1][:3])))
            else: actions.append(line)
    return actions[:20], obj if isinstance(obj,dict) else {}

class Harness:
    def __init__(self,ctx):
        self.ctx=ctx
        self.obs={}
        self.history=[]
        self.memo=''
        self.policy={'mine':{'coal':20},'craft':{},'feed':{},'collect':True,'fuel':True}
        self.calls=0
        self.bad=0
        self.recipes={}
        self.no_more_models=False

    def update(self,o):
        self.obs=unpack(o)
        for r in self.obs.get('enabled_recipes',[]):
            if isinstance(r,dict): self.recipes[r['name']]=r

    def entities(self):
        return self.obs.get('entities',[])

    def resolve(self,s):
        def sub(m):
            name,idx=m.group(1),int(m.group(2))
            es=sorted([e for e in self.entities() if e.get('name')==name],key=lambda e:e['id'])
            if idx>=len(es): raise ValueError('Unresolved entity '+m.group())
            return str(es[idx]['id'])
        return re.sub(r'@([\w-]+):(\d+)',sub,s)

    def set_policy(self,key,value):
        def clean_targets(d):
            if not isinstance(d,dict): return {}
            return {k:max(0,min(100,int(v))) for k,v in list(d.items())[:30]
                    if isinstance(k,str) and re.fullmatch(NAME,k) and
                    isinstance(v,(int,float)) and (isinstance(v,int) or math.isfinite(v))}
        if key in ['mine','craft']:
            if isinstance(value,dict): self.policy[key]=clean_targets(value)
        elif key=='feed' and isinstance(value,dict):
            self.policy[key]={ref:clean_targets(items) for ref,items in list(value.items())[:40]
                              if isinstance(ref,str) and re.fullmatch(REF,ref) and isinstance(items,dict)}
        elif key in ['fuel','collect'] and isinstance(value,bool): self.policy[key]=value

    def resource(self,name):
        rows=[r for r in self.obs.get('resources',[]) if len(r)>=5 and r[1]==name and r[4]>0]
        return max(rows,key=lambda r:r[4]) if rows else None

    def queue_ticks(self):
        return sum(max(0,int(j.get('count',1))-1)*int(j.get('ticks',0))+int(j.get('remaining',j.get('ticks',0))) for j in self.obs.get('queue',[]) if isinstance(j,dict))

    def service(self,manual=True,explicit=''):
        """Compile observed inventory deficits into legal, quota-bounded operations."""
        stock=dict(self.obs.get('stock',{})); quota=100; ops=[]
        def transfer(verb,e,role,item,count):
            nonlocal quota
            n=max(0,min(int(count),quota))
            if verb=='insert': n=min(n,int(stock.get(item,0)))
            if n and len(ops)<25:
                ops.append(f'{verb} {e["id"]} {role} {item} {n}')
                stock[item]=stock.get(item,0)+(n if verb=='extract' else -n); quota-=n
        # Explicit transfers own this turn's quota, avoiding double accounting.
        do_transfers=not re.search(r'\b(?:insert|extract|mine_building|set_recipe)\b',explicit)
        if do_transfers:
            if self.policy.get('collect',True):
                for e in self.entities():
                    for role in ['result','output']:
                        for item,n in inv(e,role).items(): transfer('extract',e,role,item,min(n,40))
            if self.policy.get('fuel',True):
                for e in self.entities():
                    if e.get('name') in ['burner-mining-drill','stone-furnace','steel-furnace','boiler','burner-inserter']:
                        target=8 if e.get('name')=='boiler' else 4
                        fuels=inv(e,'fuel'); have=sum(fuels.values())
                        if have<2 and (not fuels or 'coal' in fuels):
                            transfer('insert',e,'fuel','coal',max(0,target-have))
            feeds=self.policy.get('feed',{})
            if isinstance(feeds,dict):
                for ref,items in feeds.items():
                    try: eid=int(self.resolve(ref))
                    except (ValueError,TypeError): continue
                    e=next((e for e in self.entities() if e.get('id')==eid),None)
                    if not e or not isinstance(items,dict): continue
                    role='source' if 'furnace' in e.get('name','') else ('lab' if e.get('name')=='lab' else 'input')
                    for item,target in items.items():
                        if isinstance(target,(int,float)): transfer('insert',e,role,item,max(0,min(100,target)-inv(e,role).get(item,0)))
            for e in self.entities():
                if e.get('name')=='lab':
                    for item in ['automation-science-pack','logistic-science-pack']:
                        transfer('insert',e,'lab',item,max(0,15-inv(e,'lab').get(item,0)))
        # A completed research otherwise leaves every lab idle between calls.
        research=self.obs.get('research',{})
        if not research.get('current') and not research.get('name') and 'set_research' not in explicit:
            labs=[e for e in self.entities() if e.get('name')=='lab']
            supplied={item for item,n in stock.items() if n>0 and item.endswith('science-pack')}
            for lab in labs: supplied.update(k for k,v in inv(lab,'lab').items() if v>0)
            candidates=[]
            for tech in self.obs.get('available_research',[]):
                ingredients=tech.get('ingredients',[])
                if not isinstance(ingredients,list) or not ingredients: continue
                names={i.get('name') for i in ingredients if isinstance(i,dict)}
                prereqs=tech.get('prerequisites',[])
                if names and names<=supplied and not tech.get('researched') and all(p.get('researched') for p in prereqs if isinstance(p,dict)):
                    candidates.append(tech)
            if labs and candidates:
                tech=min(candidates,key=lambda t:(t.get('name')!='automation',t.get('seconds',1000),t.get('units',1000)))
                ops.append('set_research '+tech['name'])
        # Queued handcrafts reserve inputs. Never recursively spend future outputs.
        if manual and not self.obs.get('queue'):
            coal=self.resource('coal')
            emergency=bool(coal and stock.get('coal',0)<8 and self.policy.get('fuel',True))
            if emergency: ops.append(f'mine_resource {coal[0]} 15')
            goals=self.policy.get('craft',{})
            if isinstance(goals,dict) and not emergency:
                for item,target in goals.items():
                    if not isinstance(target,(int,float)) or stock.get(item,0)>=target: continue
                    action=self.craft_deficit(item,min(int(target),100),stock,set())
                    if action:
                        ops.append(action); break
            if not any(op.startswith(('handcraft ','mine_resource ','mine_tree ')) for op in ops):
                goals=self.policy.get('mine',{})
                if isinstance(goals,dict):
                    for item,target in goals.items():
                        if not isinstance(target,(int,float)) or stock.get(item,0)>=target: continue
                        r=self.resource(item)
                        if r:
                            n=min(15,int(target-stock.get(item,0)),int(r[4]))
                            if n>0: ops.append(f'mine_resource {r[0]} {n}'); break
                        elif item=='wood' and self.obs.get('trees'):
                            ops.append(f'mine_tree {self.obs["trees"][0][0]}'); break
        return ops

    def craft_deficit(self,item,target,stock,seen):
        if item in seen: return None
        seen=set(seen); seen.add(item)
        r=self.recipes.get(item)
        if not r or r.get('category') not in ['crafting','basic-crafting']: return None
        ingredients=r.get('ingredients',[])
        if any(i.get('type')=='fluid' for i in ingredients): return None
        output=sum(p.get('amount',1) for p in r.get('products',[]) if p.get('name')==item)
        if not output: return None
        n=max(0,math.ceil((target-stock.get(item,0))/output))
        n=min(n,max(1,int(29/max(float(r.get('seconds',.5)),.01))))
        for i in ingredients:
            need=int(i.get('amount',1))*n
            if stock.get(i['name'],0)<need:
                child=self.craft_deficit(i['name'],need,stock,seen)
                if child: return child
        for i in ingredients: n=min(n,int(stock.get(i['name'],0)//i.get('amount',1)))
        return f'handcraft {item} {n}' if n>0 else None

    def prerequisite(self,line):
        """Repair an explicit craft's missing intermediate, never future outputs."""
        if ';' in line or self.obs.get('queue'): return None
        words=line.split()
        stock=self.obs.get('stock',{})
        if len(words)==3 and words[0]=='handcraft':
            r=self.recipes.get(words[1]); count=int(words[2])
            if not r or count>100: return None
            for ingredient in r.get('ingredients',[]):
                name=ingredient['name']; need=ingredient.get('amount',1)*count
                if ingredient.get('type','item')=='item' and stock.get(name,0)<need:
                    action=self.craft_deficit(name,int(need),stock,set())
                    if action: return action
        elif len(words) in [4,5] and words[0]=='place' and stock.get(words[1],0)<1:
            return self.craft_deficit(words[1],1,stock,set())
        return None

    def clamp_transfers(self,line):
        stock=dict(self.obs.get('stock',{})); quota=100; result=[]
        cargo={(e['id'],role):dict(inv(e,role)) for e in self.entities()
               for role in ['fuel','source','result','input','output','lab','chest']}
        for op in line.split(';'):
            words=op.strip().split()
            if not words: continue
            if words[0] in ['insert','extract'] and len(words)==5:
                verb,eid,role,item,want=words
                inventory=cargo.get((int(eid),role))
                if inventory is None:
                    result.append(op.strip()); continue
                n=min(int(want),quota,stock.get(item,0) if verb=='insert' else inventory.get(item,0))
                if n<=0: continue
                stock[item]=stock.get(item,0)+(n if verb=='extract' else -n)
                inventory[item]=inventory.get(item,0)+(n if verb=='insert' else -n)
                quota-=n
                result.append(f'{verb} {eid} {role} {item} {n}')
            else:
                result.append(op.strip())
                # Account for earlier reservations in the same batch.
                if words[0]=='handcraft' and len(words)==3:
                    recipe=self.recipes.get(words[1],{})
                    for ingredient in recipe.get('ingredients',[]):
                        item=ingredient['name']
                        stock[item]=max(0,stock.get(item,0)-ingredient.get('amount',1)*int(words[2]))
                elif words[0]=='place' and len(words)>=4:
                    stock[words[1]]=max(0,stock.get(words[1],0)-1)
        return '; '.join(result) or 'wait'

    def step(self,line,auto=True):
        try: line=self.resolve(line)
        except ValueError as e:
            self.history.append({'skipped':line,'error':str(e)}); return False
        completed=self.obs.get('research',{}).get('completed',{})
        triggers={t.get('name') for t in self.obs.get('available_research',[]) if t.get('trigger')}
        filtered=[]
        for operation in line.split(';'):
            words=operation.strip().split()
            if len(words)==2 and words[0]=='set_research' and (words[1] in completed or words[1] in triggers):
                continue
            filtered.append(operation.strip())
        line='; '.join(filtered) or 'wait'
        if re.search(r'\b(?:insert|extract)\b',line): line=self.clamp_transfers(line)
        verb=line.split()[0] if line else 'wait'
        readonly=verb in READONLY
        if line=='service': line=''
        ops=[]
        if auto and not readonly:
            manual=not re.search(r'\b(?:handcraft|mine_resource|mine_tree)\b',line)
            ops=self.service(manual, line)
        native=[x.strip() for x in line.split(';')] if line and line!='wait' else []
        # User/model operations always retain their arguments and order. Defer
        # maintenance if the native batch already uses most of the 32 slots.
        ops=ops[:max(0,32-len(native))]+native
        action='; '.join(ops) or 'wait'
        before=self.obs.get('turn',0)
        try:
            result=self.ctx.act(action)
            self.update(result if result is not None else self.ctx.observe())
        except Exception as e:
            if self.ctx.done: return False
            self.history.append({'action':action,'exception':str(e)[:300]}); return False
        results=self.obs.get('results',[])
        failed=any(isinstance(r,dict) and r.get('ok') is False for r in results)
        self.history.append({'turn':self.obs.get('turn'),'action':action,'results':results,'score':self.obs.get('score')})
        self.history=self.history[-16:]
        return not failed and (readonly or self.obs.get('turn',0)>before)

    def bootstrap(self):
        # A direct-fed first iron furnace; all coordinates are derived from this map.
        patches=self.obs.get('patches',[])
        p=next((p for p in patches if p.get('name')=='iron-ore'),None)
        coal=self.resource('coal'); stone=self.resource('stone'); copper=self.resource('copper-ore')
        if not p or not coal or not stone or not copper: return
        x=int(p['center']['x']); y=int(p['center']['y'])
        actions=[f'place burner-mining-drill {x} {y} 4; place stone-furnace {x+2} {y}; mine_resource {coal[0]} 15',
          f'insert @burner-mining-drill:0 fuel coal 5; insert @stone-furnace:0 fuel coal 5; mine_resource {stone[0]} 10',
          f'handcraft stone-furnace 1; mine_resource {copper[0]} 14',
          'place stone-furnace 0 0; handcraft iron-gear-wheel 3',
          'insert @stone-furnace:1 source copper-ore 14; insert @stone-furnace:1 fuel coal 3']
        for a in actions:
            if self.ctx.done or not self.step(a,auto=False): break
        self.policy['feed']={'@stone-furnace:1':{'copper-ore':20}}
        self.policy['mine']={'coal':20,'copper-ore':25,'stone':15,'wood':12}
        if self.obs.get('turn',0)>=5 and self.obs.get('production',{}).get('iron-plate',0)>0:
            while not self.ctx.done and self.obs.get('turn',0)<12:
                if not self.step('service'): break

    def summary(self):
        o=self.obs
        out={k:v for k,v in o.items() if k not in ['resources','trees','water_rows','enabled_recipes','baseline_production','baseline_consumption','manual_output','seed','evaluation_seed']}
        reps={}
        for name in ['iron-ore','copper-ore','coal','stone']:
            rows=[r for r in o.get('resources',[]) if r[1]==name]
            p=next((p for p in o.get('patches',[]) if p.get('name')==name),{})
            c=p.get('center',{'x':0,'y':0})
            rows.sort(key=lambda r:(r[2]-c['x'])**2+(r[3]-c['y'])**2)
            reps[name]=rows[:5]
        out['resource_samples']=reps
        out['trees']=o.get('trees',[])[:12]
        out['recipes']={r['name']:{'in':{i['name']:i.get('amount',1) for i in r.get('ingredients',[])},'out':{p['name']:p.get('amount',1) for p in r.get('products',[])},'seconds':r.get('seconds'),'category':r.get('category')} for r in o.get('enabled_recipes',[]) if isinstance(r,dict)}
        out['aliases']={}
        counts={}
        for e in sorted(self.entities(),key=lambda e:e.get('id',0)):
            n=e.get('name',''); idx=counts.get(n,0); counts[n]=idx+1
            out['aliases'][f'@{n}:{idx}']=e.get('id')
        # Water is only relevant near the power block; preserve exact local coastline.
        pump=o.get('pump_site',{}); px=pump.get('x',0); py=pump.get('y',0)
        out['water_near_pump']=[{'y':r['y'],'runs':[[max(a,math.floor(px)-20),min(b,math.floor(px)+20)] for a,b in r['runs'] if b>=px-20 and a<=px+20]} for r in o.get('water_rows',[]) if abs(r['y']-py)<=22]
        out['worker_remaining_ticks']=self.queue_ticks()
        return out

    def think_with_reserve(self,prompt,budget):
        # Backend admission may itself be slow. Keep a separate wall reserve for
        # the actual factory, and leave cancellation of live inference to ctx.
        future=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,
            timeout_s=min(210.,max(30.,budget['wall_s']-100))))
        while not self.ctx.done:
            left=self.ctx.budget.remaining().get('wall_s',0)
            reserve=max(200.,2.5*self.obs.get('remaining_turns',128)+35.)
            if left<=reserve:
                self.no_more_models=True
                raise TimeoutError('Factory finishing reserve reached')
            try:
                return future.result(timeout=min(15.,max(.1,left-reserve)))
            except TimeoutError:
                if future.done(): return future.result()
                # Only early smelting is advanced speculatively. Do not consume
                # the whole science horizon while a design is in flight.
                if self.obs.get('turn',0)<12: self.step('service')
        self.no_more_models=True
        raise TimeoutError('Episode closed while planning')

    def run_episode(self,game):
        self.update(self.ctx.observe())
        if not self.obs: return
        self.bootstrap()
        plan=[]; queries=0; repair_counts={}
        while not self.ctx.done:
            budget=self.ctx.budget.remaining()
            if not plan and budget.get('system2_calls',0)>1 and budget.get('wall_s',0)>160 and self.calls<17 and self.bad<4 and not self.no_more_models:
                prompt=INSTRUCTIONS+'\nGAME RULES:\n'+game.describe()+'\nSTATE:\n'+json.dumps(self.summary(),separators=(',',':'))+'\nPOLICY:\n'+json.dumps(self.policy)+'\nMEMO:\n'+self.memo+'\nRECENT ACTIONS:\n'+json.dumps(self.history[-10:],separators=(',',':'))+'\nBUDGET:\n'+json.dumps(budget)
                try:
                    response=self.think_with_reserve(prompt,budget)
                    self.calls+=1
                    plan,obj=parse_reply(response.text)
                    if isinstance(obj.get('memo'),str): self.memo=obj['memo'][:2400]
                    if isinstance(obj.get('policy'),dict):
                        for k in ['mine','craft','feed','collect','fuel']:
                            if k in obj['policy']: self.set_policy(k,obj['policy'][k])
                    self.bad=0 if plan else self.bad+1
                except Exception as e:
                    if self.ctx.done: break
                    self.calls+=1; self.bad+=1
                    self.history.append({'model_error':type(e).__name__})
                if not plan: plan=['service']*3
            if not plan:
                if 'automation-science-pack' in self.recipes and any(e.get('name')=='lab' for e in self.entities()):
                    goals=self.policy.get('craft',{})
                    if not any(k.endswith('science-pack') and v>0 for k,v in goals.items()):
                        self.policy['craft']={'automation-science-pack':15}
                plan=['service']*8
            line=plan.pop(0)
            repair=self.prerequisite(line) if repair_counts.get(line,0)<6 else None
            if repair:
                repair_counts[line]=repair_counts.get(line,0)+1
                self.history.append({'repair_for':line,'prerequisite':repair})
                plan.insert(0,line)
                line=repair
            else:
                repair_counts.pop(line,None)
            if line.split()[0] in READONLY:
                queries+=1
                if queries>12: line='service'
            else: queries=0
            ok=self.step(line)
            if not ok:
                plan=[]
                self.bad+=1
                if not self.ctx.done: self.step('service')
        self.ctx.memory['factory_memo']=self.memo

INSTRUCTIONS='''You supervise a real Factorio factory for SCORE over 128 turns. Return compact JSON only:
{"memo":"brief continuing plan and facts","policy":{"mine":{"coal":25},"craft":{},"feed":{}},"actions":["full native action; full native action","next turn action","service 6"]}
Every actions entry is one turn (semicolon operations execute BEFORE 30sec simulation). Provide 12-20 productive turns per call. Model admission can take many minutes: finish a whole milestone per response, including future crafts and placements using aliases. Prefer a complete power+lab build plan once recipes are unlocked. Use up to20 turns. Do not wait for model feedback after every craft. Recipe outputs from a handcraft cannot be used until a later turn. Place buildings in one turn, refer to them on later turns with @item:INDEX, index 0-based sorted by stable ID; e.g. insert @boiler:0 fuel coal 10. Native numeric IDs also work. Do not repeat transfer boilerplate in every action: the policy already collects and feeds, and explicit transfers disable that automatic service for the turn. Do NOT guess future numeric IDs. Queries recipe/technology/inspect must be alone. No invented native commands. Complete strings preserve arguments and order; native insert/extract counts are clamped to current availability and the100-item quota, and zero transfers are omitted.
CODE SERVICES every turn unless explicit insert/extract/set_recipe/mine_building: collect up to40 output/result per building; refuel burners (4 coal; boilers8); feed configured buildings; put stock science into labs, and select an available research when current research is empty and the needed science is present (automation first, then shortest lab time). Policy feed maps entity alias -> item:desired input count, e.g. {"@assembling-machine-1:0":{"iron-gear-wheel":15,"copper-plate":15},"@stone-furnace:1":{"copper-ore":20}}. Feed furnaces source, assemblers input. NO automatic assembler feeding unless configured. Policy mine lists stock targets in priority order; craft lists desired stock targets, can craft missing intermediates from observed enabled recipes. On turns with no explicit manual action and idle worker it starts ONE missing craft or mine. Emergency coal below8 is mined first; otherwise manual craft policy has priority over mining. Clear obsolete craft goals! Script never mines plates; ore must be smelted. `service N` runs N maintenance turns with feedback each turn. `wait` also receives maintenance. Explicit manual actions suppress automatic manual work that turn. Single handcraft/place operations missing craftable intermediates receive up to6 automatic prerequisite turns, then retry; base resources still need real mining/smelting. New policy fields replace entire previous field; omitted fields persist. Read-only queries have no maintenance. Input/output transfers share 100 items, and manual queue max60secs; prefer <=29sec manual work per turn. Automatic mining batches are15 cycles=30sec based on observed ore work.
STRATEGY: initial direct drill->iron furnace plus manually fed copper furnace are provided if successful. Read results; repair if necessary. Bootstrap steam power (50 engine iron), electronics (10 copper), craft lab to unlock red science. Build power near validated pump_site; place pump in its exact observed orientation, connect water->boiler->steam-engine with pipes and fuel, poles cover both engine and labs/assemblers and connect within7.5tiles. Offshore pump output points toward land (opposite its direction). Correct actual fluid ports by observing fluid/status. A boiler with full_output and engine with no_input_fluid indicates disconnected steam: check for a one-tile pipe gap before replacing the machinery. Boiler is3x2 with steam output perpendicular to water throughflow; steam-engine3x5; use half tiles for odd axes. Leave room! Build labs/assemblers close to power to avoid a huge pole chain. Remote transfer allows distant smelters. Handcraft red packs and consume in powered lab first to research automation, then assemble red science for extra engine score. More red throughput is valuable; prioritize complete chains. Science research must be selected for labs to consume. automation-science-pack is a native lab-craft trigger, never a science-consuming research; select automation after crafting the lab. Commands selecting completed or trigger technologies are omitted by the supervisor. Choose available red-only tech while producing red; do not stall behind green. Green needs complete supply and enough horizon; don't chase oil. Many red-only technologies provide hundreds of lab consumptions. Each red pack engine-produced+consumed gives5, manual+consumed4. A working lab is key. Extra furnaces/direct burner drill supply and service transfers are cheap enough at modest scale. Research and unlocks grant no inventory. Keep fuel; simple direct transfers and manual crafts can outperform complex belts within this horizon. Put assembler feeds and stock craft/mining targets in policy so production continues between decisions. Engine production/consumption deltas are the evidence of success. Stored electric energy alone does not prove connected supply. Repair stalled chains before expanding. Resource samples are usable IDs, patch centers have plentiful ore; other terrain is grass except water and trees. Use observed boxes to avoid collisions. End by maximizing science consumption and maintaining production, not constructing idle capital. Keep answers concise and executable.'''

def build(ctx):
    return Harness(ctx)
