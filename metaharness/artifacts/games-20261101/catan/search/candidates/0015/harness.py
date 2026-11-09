"""Economic projects, with Astra as a sparse strategic portfolio manager.
All game knowledge comes from public observations; execution uses live legal IDs.
"""
from __future__ import annotations
import json
import heapq
import re
import time
from metaharness.core.contract import BudgetExceeded

R = ('WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE')
COST = {'build_road': (1,1,0,0,0), 'build_settlement': (1,1,1,1,0),
        'build_city': (0,0,0,2,3), 'buy_development_card': (0,0,1,1,1)}
PIPS = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}

def key(a):
    return (a['type'], json.dumps(a.get('parameters'), sort_keys=True))

def parse(text):
    if not isinstance(text,str):return {}
    for i,c in enumerate(text):
        if c == '{':
            try:
                x,_ = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(x,dict): return x
            except (ValueError,TypeError): pass
    return {}

class State:
    def __init__(self,o):
        self.o=o; self.me=o.get('self',{}); self.color=self.me.get('id')
        self.legal=o.get('legal_actions',[]); self.bytype={}
        for a in self.legal: self.bytype.setdefault(a['type'],[]).append(a)
        b=o.get('board',{}); self.v={x['id']:x for x in b.get('vertices',[])}
        self.h={x['id']:x for x in b.get('hexes',[])}
        self.edges={tuple(sorted(x['id'])):x.get('owner') for x in b.get('edges',[])}
        self.hand=tuple(self.me.get('resources',{}).get(r,0) for r in R)
        self.bank=tuple(o.get('bank_counts',{}).get(r,19) for r in R)
        ports=self.me.get('ports',[])
        self.rate=tuple(2 if r in ports else 3 if 'ANY' in ports else 4 for r in R)
        self.prod={c:[0.0]*5 for c in [self.color]+[p['id'] for p in o.get('opponents',[])]}
        self.raw={c:[0.0]*5 for c in self.prod}
        self.site={}; self.mine=[]; self.settlements=[]
        for n,v in self.v.items():
            p=[0]*5
            for hi in v.get('adjacent_hexes',[]):
                h=self.h.get(hi,{})
                if h.get('terrain') in R: p[R.index(h['terrain'])]+=PIPS.get(h.get('number'),0)
            self.site[n]=p
            c=v.get('owner')
            if c==self.color:
                self.mine.append(n)
                if v.get('building')=='SETTLEMENT': self.settlements.append(n)
            if c in self.prod:
                mult=2 if v.get('building')=='CITY' else 1
                for hi in v.get('adjacent_hexes',[]):
                    h=self.h.get(hi,{})
                    if h.get('terrain') not in R: continue
                    j=R.index(h['terrain']); amount=mult*PIPS.get(h.get('number'),0)/9.0
                    self.raw[c][j]+=amount
                    self.prod[c][j]+=amount*(0.25 if h.get('robber') else 1.0)
        self.production=self.prod.get(self.color,[0.0]*5)
        self.vp=self.me.get('true_vp',0)
        self.setup='initial' in str(o.get('phase',''))
        self.own_edges=[e for e,c in self.edges.items() if c==self.color]
        self.sources=set(self.mine)
        for e in self.own_edges: self.sources.update(n for n in e if self.v[n].get('owner') in (None,self.color))
        self.routes=self.find_routes()

    def open_site(self,n):
        v=self.v[n]
        return not v.get('owner') and all(not self.v[x].get('owner') for x in v.get('neighbors',[]))

    def find_routes(self, extra=()):
        edges=dict(self.edges)
        sources=set(self.sources)
        for e in extra:
            edges[tuple(sorted(e))]=self.color; sources.update(e)
        best={n:(0,[]) for n in sources if self.v[n].get('owner') in (None,self.color)}
        q=[(0,n,[]) for n in best]; heapq.heapify(q)
        while q:
            d,n,path=heapq.heappop(q)
            if d!=best[n][0] or d>=4: continue
            if self.v[n].get('owner') not in (None,self.color): continue
            for z in self.v[n].get('neighbors',[]):
                e=tuple(sorted((n,z))); owner=edges.get(e)
                if owner not in (None,self.color): continue
                nd=d+(owner!=self.color)
                if nd < best.get(z,(99,[]))[0]:
                    pp=path+([list(e)] if owner!=self.color else [])
                    best[z]=(nd,pp); heapq.heappush(q,(nd,z,pp))
        return {n:p for n,(d,p) in best.items() if self.open_site(n) and d<=4}

    def pip_value(self,n):
        p=self.site.get(n,[0]*5); prod=self.raw.get(self.color,[0]*5)
        weights=[1.0,0.95,1.0,1.35,1.45]
        # Scarce resources are valuable, but not infinitely so; ports substitute.
        val=sum(x*w*(1.25 if prod[i]<0.3 else 1) for i,(x,w) in enumerate(zip(p,weights)))
        port=self.v.get(n,{}).get('port')
        if port=='ANY': val+=1.5
        elif port in R: val+=min(3,prod[R.index(port)]*1.7)
        return val

    def opening_value(self,n):
        p=self.site[n]; old=self.raw.get(self.color,[0]*5)
        combo=[old[i]*9+p[i] for i in range(5)]
        val=sum(p[i]*[1,0.9,1.0,1.4,1.5][i] for i in range(5))
        val+=sum(min(5,x)*[0.7,0.6,0.8,1.1,1.2][i] for i,x in enumerate(combo))
        if old and sum(old)>0:
            val-=sum(4 for i in (3,4) if combo[i]==0)
            val-=sum(1.5 for i in (0,1,2) if combo[i]==0)
        val+=0.25*sum(1 for z in self.v[n].get('neighbors',[]) for y in self.v[z].get('neighbors',[]) if y!=n and self.open_site(y))
        port=self.v[n].get('port')
        if port=='ANY': val+=1.4
        elif port in R: val+=min(4,combo[R.index(port)]*0.4)
        return val

    def eta(self,cost,hand=None):
        hand=self.hand if hand is None else hand
        # Expected-resource relaxation, in our rounds. A ranking, not a promise.
        for k in range(121):
            t=k/4
            h=[hand[i]+t*self.production[i] for i in range(5)]
            missing=sum(max(0,cost[i]-h[i]) for i in range(5))
            credit=sum(max(0,h[i]-cost[i])/self.rate[i] for i in range(5))
            if credit+1e-8>=missing: return t
        return 35.0

    def trades(self,cost,hand=None):
        """Exact no-waste trade sequence preserving the complete project's cost.
        Only deficit resources are bought; a resource is never traded away then bought.
        The authoritative menu checks the first step on every invocation.
        """
        hand=self.hand if hand is None else tuple(hand)
        def rec(h,b,path):
            deficits=[i for i in range(5) if h[i]<cost[i]]
            if not deficits: return path
            if len(path)>=9: return None
            j=min(deficits,key=lambda j:b[j])
            if b[j]<=0: return None
            options=sorted((i for i in range(5) if h[i]-cost[i]>=self.rate[i]),key=lambda i:(self.rate[i],-h[i]))
            for i in options:
                hh=list(h); bb=list(b); rate=self.rate[i]
                hh[i]-=rate; hh[j]+=1; bb[i]+=rate; bb[j]-=1
                result=rec(tuple(hh),tuple(bb),path+[(i,j,rate)])
                if result is not None: return result
            return None
        return rec(tuple(hand),self.bank,[])

    def road_length(self,extra=()):
        es=list(dict.fromkeys(self.own_edges+[tuple(sorted(e)) for e in extra]))
        # Observation-verified conservative bound: an edge into an enemy
        # building did not increase the engine award length in dev play.
        es=[e for e in es if all(self.v[n].get('owner') in (None,self.color) for n in e)]
        adj={}
        for i,(a,b) in enumerate(es):
            adj.setdefault(a,[]).append((b,i)); adj.setdefault(b,[]).append((a,i))
        def dfs(n,mask):
            if mask and self.v[n].get('owner') not in (None,self.color): return 0
            return max([0]+[1+dfs(z,mask|(1<<i)) for z,i in adj.get(n,[]) if not mask&(1<<i)])
        return max([0]+[dfs(n,0) for n in adj])

    def award_routes(self):
        owner=self.o.get('awards',{}).get('longest_road_owner')
        if owner==self.color or len(self.own_edges)<3: return []
        needed=max(5,self.o.get('awards',{}).get('longest_road_length',0)+1)
        now=self.road_length()
        if needed-now>3: return []
        beam=[[]]; seen=set(); winners=[]
        for depth in range(1,min(3,self.me.get('pieces',{}).get('roads',15))+1):
            nxt=[]
            for path in beam:
                used=set(self.own_edges+[tuple(e) for e in path]); nodes=set(self.sources)
                for e in path: nodes.update(e)
                for n in nodes:
                    if self.v[n].get('owner') not in (None,self.color): continue
                    for z in self.v[n].get('neighbors',[]):
                        e=tuple(sorted((n,z)))
                        if e in used or self.edges.get(e) is not None: continue
                        pp=path+[e]; sig=tuple(sorted(pp))
                        if sig in seen: continue
                        seen.add(sig); length=self.road_length(pp)
                        if length>=needed: winners.append(pp)
                        else: nxt.append((length,pp))
            if winners: return winners[:12]
            nxt.sort(key=lambda x:x[0],reverse=True); beam=[p for _,p in nxt[:35]]
        return []

    def projects(self,mode='balanced',preferred=None):
        out=[]; pieces=self.me.get('pieces',{})
        def add(pid,typ,target,path,cost,value):
            t=self.eta(cost); trades=self.trades(cost)
            score=value/(1.7+t)
            if mode=='cities' and typ=='build_city': score*=1.18
            if mode=='expand' and typ=='build_settlement': score*=1.18
            if mode=='army' and typ=='buy_development_card': score*=1.35
            if pid==preferred: score*=1.18
            if trades is not None: score*=1.12
            out.append(dict(id=pid,type=typ,target=target,path=path,cost=list(cost),eta=t,value=round(value,2),score=round(score,3),ready=trades is not None))
        late=max(0,self.vp-5)
        if pieces.get('cities',4)>0:
            for n in self.settlements:
                value=6+0.60*self.pip_value(n)+late*2
                if self.vp>=9:value+=200
                add('city:'+str(n),'build_city',n,[],COST['build_city'],value)
        if pieces.get('settlements',5)>0:
            for n,path in self.routes.items():
                if len(path)>min(3,pieces.get('roads',15)): continue
                c=list(COST['build_settlement']);c[0]+=len(path);c[1]+=len(path)
                value=8+0.62*self.pip_value(n)+late*2
                if self.vp>=9:value+=200
                # Further routes risk being contested before completion.
                value/=1+0.15*len(path)
                add('settle:'+str(n),'build_settlement',n,path,c,value)
        if self.o.get('development_cards_remaining',0)>0:
            army=self.o.get('awards',{}).get('largest_army_owner')
            knights=self.me.get('played_knights',0)
            army_gap=max(3,self.o.get('awards',{}).get('largest_army_size',0)+1)-knights
            val=6.0+late*1.2
            if army!=self.color and army_gap<=3: val+=2.0
            if army==self.color: val-=0.5
            if self.vp>=9: val+=8
            add('dev','buy_development_card',None,[],COST['buy_development_card'],val)
        for path in (self.award_routes() if getattr(self,'road_awards_enabled',True) else []):
            n=len(path); value=18+late*3
            if self.vp>=8: value+=220
            add('road:'+json.dumps(path),'award',None,[list(e) for e in path],[n,n,0,0,0],value)
        return sorted(out,key=lambda x:x['score'],reverse=True)

    def find(self,typ,params=None):
        for a in self.bytype.get(typ,[]):
            if params is None or a.get('parameters')==params: return a
        return None

    def first_trade(self,triples):
        if not triples: return None
        i,j,rate=triples[0]
        for a in self.bytype.get('maritime_trade',[]):
            p=a.get('parameters',[])
            if isinstance(p,list) and len(p)>=2 and p[-1]==R[j] and [x for x in p[:-1] if x is not None]==[R[i]]*rate: return a
        return None

    def project_action(self,p):
        path=p.get('path',[])
        if path: return self.find('build_road',path[0])
        return self.find(p['type'],p.get('target'))


# Counterfactual valuation, deliberately separate from the exact action compiler.
BaseState = State

def flow_time(demand, production, rates):
    """Exact continuous exchange relaxation for a long-run resource budget.

    Piecewise-linear surplus/rate >= deficits. NOT discrete build readiness.
    No dice are sampled and no synthetic state is ever passed to ctx.act.
    """
    if sum(max(0,-d)/r-max(0,d) for d,r in zip(demand,rates))>=-1e-10:return 0.0
    cuts=sorted({0.0} | {max(0.0,d/p) for d,p in zip(demand,production) if p>1e-9})
    for low,high in zip(cuts,cuts[1:]+[float('inf')]):
        mid=low+1e-7
        weights=[1.0/rates[i] if mid*production[i]>=demand[i] else 1.0 for i in range(5)]
        slope=sum(weights[i]*production[i] for i in range(5))
        if slope<=1e-12:continue
        root=sum(weights[i]*demand[i] for i in range(5))/slope
        if low-1e-6<=root<=high+1e-6:return max(0.0,root)
    return 1000.0

class VictoryEconomy:
    """Observed geometry, resource demand and explicit uncertain card priors."""
    def __init__(self,s):
        self.s=s
        self.production=list(s.raw.get(s.color,[0]*5))
        self.rates=list(s.rate)
        self.vp=float(s.vp)
        self.settlements=len(s.settlements)
        self.cities=len(s.mine)-self.settlements
        self.knights=float(s.me.get('played_knights',0)+s.me.get('development_cards',{}).get('KNIGHT',0))
        aw=s.o.get('awards',{})
        self.army_owned=aw.get('largest_army_owner')==s.color
        self.army_target=max(3,aw.get('largest_army_size',0)+1)
        self.road_owned=aw.get('longest_road_owner')==s.color
        self.deck=s.o.get('development_cards_remaining',0)
        self.roads_left=s.me.get('pieces',{}).get('roads',15)
        self.paths={n:[tuple(e) for e in path] for n,path in s.routes.items()}
        self.award_cost=min([len(p) for p in s.award_routes()] or [99]) if getattr(s,'road_awards_enabled',True) else 99

    def clone(self):
        import copy
        z=copy.copy(self)
        z.production=list(self.production);z.rates=list(self.rates);z.paths=dict(self.paths)
        return z

    def after(self,p):
        z=self.clone();typ=p['type'];n=p.get('target')
        if typ in ('build_city','build_settlement'):
            z.vp+=1
            z.production=[x+y/9 for x,y in zip(z.production,self.s.site[n])]
            if typ=='build_city':z.settlements-=1;z.cities+=1
            else:
                z.settlements+=1
                paid={tuple(sorted(e)) for e in p['path']}
                z.roads_left-=len(paid)
                forbidden={n}|set(self.s.v[n].get('neighbors',[]))
                z.paths={v:[e for e in path if tuple(sorted(e)) not in paid]
                         for v,path in z.paths.items() if v not in forbidden}
                port=self.s.v[n].get('port')
                if port=='ANY':z.rates=[min(r,3) for r in z.rates]
                elif port in R:z.rates[R.index(port)]=2
        elif typ=='award':
            z.vp+=2;z.road_owned=True;z.roads_left-=len(p['path'])
        return z

    def frontier(self):
        # Independent sites: do not promise adjacent settlements in one recipe.
        chosen=[];blocked=set()
        for n,path in sorted(self.paths.items(),key=lambda x:(len(x[1]),-sum(self.s.site[x[0]]))):
            if n in blocked:continue
            chosen.append((n,len(path)))
            blocked.add(n);blocked.update(self.s.v[n].get('neighbors',[]))
            if len(chosen)>=4:break
        return chosen

    def recipes(self,hand=None):
        if self.vp>=10:return [dict(rounds=0.0,settlements=0,cities=0,cards=0,army=False,road=False,cost=[0]*5)]
        sites=self.frontier();out=[]
        army_opts=[0] if self.army_owned else [0,1]
        road_opts=[0,1] if not self.road_owned and self.award_cost<=3 else [0]
        for new_s in range(len(sites)+1):
            access=sum(x[1] for x in sites[:new_s])
            for new_c in range(min(4-self.cities,self.settlements+new_s)+1):
                if self.settlements+new_s-new_c>5:continue
                for army in army_opts:
                    for road in road_opts:
                        roads=access+road*self.award_cost
                        if roads>self.roads_left:continue
                        points=new_s+new_c+2*army+2*road
                        vp_need=max(0.0,10-self.vp-points)
                        # Published base-deck prior, NOT knowledge of remaining identities.
                        cards=max(5.0*vp_need,max(0,self.army_target-self.knights)/0.56 if army else 0)
                        if cards>self.deck+1e-6:continue
                        demand=[new_s+roads,new_s+roads,new_s+cards,new_s+2*new_c+cards,3*new_c+cards]
                        net=[demand[i]-(hand[i] if hand is not None else 0) for i in range(5)]
                        rounds=flow_time(net,self.production,self.rates)
                        # Knights must deploy one per personal turn. Current hand ages are
                        # uncertain for future plans, so one round per needed play is conservative.
                        if army:rounds=max(rounds,self.army_target-self.s.me.get('played_knights',0))
                        out.append(dict(rounds=round(rounds,4),settlements=new_s,cities=new_c,
                                        cards=round(cards,2),army=bool(army),road=bool(road),cost=demand,
                                        sites=[x[0] for x in sites[:new_s]]))
        if not out:
            # Expected-VP feasibility can be empty in a depleted deck. Keep an explicit
            # optimistic card-tail alternative; legal actions still govern execution.
            cost=[0,0,max(1,self.deck),max(1,self.deck),max(1,self.deck)]
            net=[cost[i]-(hand[i] if hand is not None else 0) for i in range(5)]
            out=[dict(rounds=flow_time(net,self.production,self.rates)+20,settlements=0,cities=0,
                      cards=self.deck,army=False,road=False,cost=cost,uncertain_tail=True)]
        return sorted(out,key=lambda p:p['rounds'])

    def time(self,hand=None):return self.recipes(hand)[0]['rounds']

class State(BaseState):
    def projects(self,mode='balanced',preferred=None):
        projects=super().projects(mode,preferred)
        econ=VictoryEconomy(self);recipes=econ.recipes();before=recipes[0]['rounds']
        self.victory_recipes=recipes[:5]
        self.economy_before=before
        wealth_before=econ.time(self.hand)
        for p in projects:
            typ=p['type']
            if typ=='buy_development_card':
                vp=econ.clone();vp.vp+=1;vp.deck=max(0,vp.deck-1)
                knight=econ.clone();knight.knights+=1;knight.deck=max(0,knight.deck-1)
                # Utility cards are credited a modest one-round saved resource budget.
                after=.20*vp.time()+.56*knight.time()+.24*max(0,before-1)
            else:after=econ.after(p).time()
            gain=max(.05,before-after)
            score=gain/(1.5+p['eta'])
            if mode=='cities' and typ=='build_city':score*=1.18
            if mode=='expand' and typ=='build_settlement':score*=1.18
            if mode=='army' and typ=='buy_development_card':score*=1.18
            if p['id']==preferred:score*=1.12
            if p['ready']:score*=1.12
            if (typ in ('build_city','build_settlement') and self.vp>=9) or (typ=='award' and self.vp>=8):score+=1000/(1+p['eta'])
            p['old_value']=p['value']
            p['value']=round(gain,3);p['score']=round(score,4)
            p['economy_rounds_after']=round(after,2)
            p['economic_gain']=round(before-after,3)
            if p['ready']:
                residual=list(self.hand)
                for i,j,rate in self.trades(p['cost']) or []:
                    residual[i]-=rate;residual[j]+=1
                residual=[x-c for x,c in zip(residual,p['cost'])]
                if typ=='buy_development_card':
                    utility=econ.clone();utility.deck=max(0,utility.deck-1)
                    net_after=.20*vp.time(residual)+.56*knight.time(residual)+.24*max(0,utility.time(residual)-1)
                else:net_after=econ.after(p).time(residual)
                p['net_gain_after_cost']=round(wealth_before-net_after,4)
                p['residual_hand']=residual
        return sorted(projects,key=lambda p:p['score'],reverse=True)


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx; self.mode='balanced'; self.preferred=None; self.memo=''
        self.calls=0; self.last_call_turn=-999; self.last_vp=-1; self.last_threat=0
        self.failure=None; self.rejects=0; self.last_score_turn=0; self.last_score=-1
        self.bad=set(); self.last_revision=None
        self.pending=None; self.pending_revision=None
        self.active_project=None; self.active_turn=None; self.road_awards_enabled=True
        self.investment=None; self.max_step_s=0.0; self.retired=[]

    def ask(self,s,projects,reason,opening=False):
        ctx=self.ctx; budget=ctx.budget.remaining()
        self.retired=[f for f in self.retired if not f.done()]
        if self.pending is not None:
            if self.failure and not self.retired and not self.pending.done():
                self.retired.append(self.pending);self.pending=None
                ctx.log('superseded_planner',reason=self.failure,old_revision=self.pending_revision)
            else:return self.poll(s,projects)
        if self.calls>=12 or budget.get('system2_calls',0)<=1 or budget.get('wall_s',0)<380: return None
        cities=[dict(vertex=n,pips=s.site[n],port=s.v[n].get('port')) for n in s.mine]
        options=[]
        if opening:
            for a in s.legal:
                n=a.get('parameters')
                if a['type']=='build_settlement': options.append(dict(**a,pips=s.site[n],heuristic=round(s.opening_value(n),2),port=s.v[n].get('port')))
            options.sort(key=lambda x:x['heuristic'],reverse=True)
        payload={'reason':reason,'phase':s.o.get('phase'),'turn':s.o.get('turn_number'),
                 'self':s.me,'opponents':s.o.get('opponents'),'awards':s.o.get('awards'),
                 'production_per_round':dict(zip(R,[round(x,2) for x in s.production])),
                 'opponent_production':{c:[round(x,2) for x in p] for c,p in s.raw.items() if c!=s.color},
                 'resource_order':R,'our_sites':cities,'mode':self.mode,'prior_plan':self.memo,
                 'progress':ctx.progress(),'failure':self.failure,
                 'projects':projects[:12],'opening_choices':options,
                 'remaining_victory_recipes':getattr(s,'victory_recipes',[]),
                 'economic_clock_rounds':getattr(s,'economy_before',None),
                 'economic_assumptions':'Long-run raw production and fractional maritime flow; no future compounding or enemy interference. Card odds use base-deck priors, not known remaining cards. Immediate ready flag uses exact integer funding.',
                 'recent_events':s.o.get('events',[])[-12:]}
        if not opening:
            payload['legal_actions']=s.legal
            payload['expansion_sites']=[{'vertex':n,'pips':s.site[n],'path':path,'port':s.v[n].get('port')} for n,path in sorted(s.routes.items(),key=lambda x:s.pip_value(x[0]),reverse=True)[:10]]
        prompt=("You manage a Catan economic planner against three strong AlphaBeta bots. Reach 10 true VP first. "
          "Choose a coherent route to victory from code-computed projects. Roads without useful endpoints are wasteful; "
          "ore/wheat city production compounds, development cards give army and hidden VP. Army incumbent wins ties. "
          "Project values measure reduction in the whole economy remaining resource budget to reach ten, not raw pips. Compare bottleneck relief and useful ports. The recipe clock is a long-run ranking, NOT a promised finish time: future compounding, dice, enemy interference and card odds are uncertain. Production/ETA are approximate expected values, not guaranteed rolls. Hidden opponent cards are unknown. "
          "Code executes forced actions, robber/discards, exact maritime trades preserving project cost, and builds, refreshing legality after each. "
          "It replans locally each step. Your project preference is a modest persistent bonus, not an unconditional build order. "
          "Use mode cities, expand, army, or balanced; project must be an offered project id (or null). "
          "For setup choose a legal action ID, consider complementary next site and grain/ore. "
          "When failure evidence is present diagnose it and revise mode/project; optionally select one current legal action ID to recover. "
          "Reply ONLY compact JSON: {\"action\":null,\"mode\":\"balanced\",\"project\":null,\"plan\":\"under 35 words\"}. "
          "No long explanation.\n"+json.dumps(payload,separators=(',',':')))
        self.calls+=1; self.last_call_turn=s.o.get('turn_number',0); self.last_vp=s.vp
        self.last_threat=max([0]+[x.get('public_vp',0) for x in s.o.get('opponents',[])])
        # Bound our own wait as well as requesting a model timeout: service
        # admission/queue delays can exceed the backend's generation timeout.
        self.pending_revision=s.o.get('revision')
        self.pending=ctx.spawn(lambda: ctx.think(prompt,session_id=None,timeout_s=300.0))
        try:
            result=self.pending.result(timeout=210.0)
            self.pending=None
            return self.accept(s,projects,result,True)
        except TimeoutError:
            ctx.log('planner_wait_limit',seconds=210,continuing_with='verified projects')
            return None
        except Exception as exc:
            self.pending=None
            ctx.log('planner_unavailable',error=str(exc)[:200])
            return None

    def accept(self,s,projects,result,allow_action):
        reply=getattr(result,'text','')
        if not isinstance(reply,str):reply=''
        ans=parse(reply)
        if ans.get('mode') in ('cities','expand','army','balanced'): self.mode=ans['mode']
        ids={p['id'] for p in projects}
        if isinstance(ans.get('project'),str) and ans['project'] in ids: self.preferred=ans['project']
        if isinstance(ans.get('plan'),str): self.memo=ans['plan'][:600]
        self.ctx.memory['strategy']={'mode':self.mode,'project':self.preferred,'plan':self.memo}
        if not allow_action:return None
        matches=re.findall(r'r\d+:a\d+',str(ans.get('action','')))
        if not matches: matches=re.findall(r'r\d+:a\d+',reply)
        for a in s.legal:
            if a['id'] in matches: return a
        return None

    def poll(self,s,projects):
        if self.pending is None or not self.pending.done():return None
        future=self.pending;self.pending=None
        try:
            result=future.result()
            return self.accept(s,projects,result,s.o.get('revision')==self.pending_revision)
        except Exception as exc:
            self.ctx.log('planner_unavailable',error=str(exc)[:200])
            return None

    def initial_road(self,s):
        baseline=s.routes
        def score(a):
            routes=s.find_routes([a['parameters']])
            improved=[(n,p) for n,p in routes.items() if len(p)<len(baseline.get(n,[0]*99))]
            return max([0]+[s.pip_value(n)/(1+len(p)) for n,p in improved])
        return max(s.bytype.get('build_road',s.legal),key=score)

    def robber(self,s):
        players={p['id']:p for p in s.o.get('opponents',[])}
        def score(a):
            coord,victim=a.get('parameters',[None,None]); h=next((h for h in s.h.values() if h.get('coordinate')==coord),{})
            val=0
            for n in h.get('vertices',[]):
                v=s.v[n]; owner=v.get('owner')
                mult=2 if v.get('building')=='CITY' else 1
                if owner==s.color: val-=3*mult
                elif owner in players:
                    p=players[owner]; threat=p.get('public_vp',0)+min(1.5,p.get('development_count',0)*0.3)
                    val+=mult*(0.8+0.12*threat+(1.0 if threat>=8 else 0))
            val*=PIPS.get(h.get('number'),0)
            if victim in players: val+=1.5+min(2,players[victim].get('resource_count',0)*0.25)
            return val
        return max(s.bytype['move_robber'],key=score)

    def choose(self,s,projects):
        b=s.bytype
        if 'move_robber' in b:return self.robber(s)
        if 'discard_resource' in b:
            goal=projects[0]['cost'] if projects else COST['build_city']
            def keep(a):
                r=a.get('parameters'); i=R.index(r) if r in R else 0
                hh=list(s.hand);hh[i]=max(0,hh[i]-1)
                return s.eta(goal,hh)+0.03*(goal[i]>hh[i])-0.001*s.hand[i]
            return min(b['discard_resource'],key=keep)
        if s.setup:
            if 'build_settlement' in b:return max(b['build_settlement'],key=lambda a:s.opening_value(a['parameters']))
            if 'build_road' in b:return self.initial_road(s)
        # Guaranteed one-point win before any heuristic.
        if s.vp>=9:
            for typ in ('build_city','build_settlement'):
                if typ in b:return max(b[typ],key=lambda a:s.pip_value(a['parameters']))
        if 'play_knight_card' in b:
            aw=s.o.get('awards',{}); knights=s.me.get('played_knights',0)
            army_gain=aw.get('largest_army_owner')!=s.color and knights+1>=max(3,aw.get('largest_army_size',0)+1)
            blocked=any(h.get('robber') and any(s.v[n].get('owner')==s.color for n in h.get('vertices',[])) for h in s.h.values())
            if army_gain or blocked or 'roll' not in b: return b['play_knight_card'][0]
        if 'roll' in b:return b['roll'][0]
        # Free road phases are not normal paid road construction.
        if set(b)=={'build_road'}:
            return self.best_road(s,projects)
        # A funded project is a transaction: finish its roads/trades/build
        # before reconsidering the portfolio. Revalidate against fresh state.
        if self.active_project is not None:
            p=next((p for p in projects if p['id']==self.active_project),None)
            if p is not None and p['ready'] and self.active_turn==s.o.get('turn_number'):
                trades=s.trades(p['cost'])
                a=s.first_trade(trades) if trades else s.project_action(p)
                if a:return a
            self.active_project=None
        # Resource cards: exploit only when they improve a real project or collect a substantial monopoly.
        if 'play_monopoly' in b:
            expected=[]
            for a in b['play_monopoly']:
                r=a.get('parameters');i=R.index(r) if r in R else 0
                amount=0
                for p in s.o.get('opponents',[]):
                    production=s.raw.get(p['id'],[0]*5)
                    amount+=p.get('resource_count',0)*(production[i]+0.25)/(sum(production)+1.25)
                expected.append((amount,a))
            amount,a=max(expected,key=lambda x:x[0])
            if amount>=3.0 or (s.vp>=8 and amount>=2):return a
        if 'play_year_of_plenty' in b and projects:
            best=None
            for a in b['play_year_of_plenty']:
                hh=list(s.hand)
                for r in a.get('parameters',[]):
                    if r in R:hh[R.index(r)]+=1
                gain=max((p['score'] if s.trades(p['cost'],hh) is not None and not p['ready'] else 0) for p in projects)
                if best is None or gain>best[0]:best=(gain,a)
            if best and best[0]>0:return best[1]
        if 'play_road_building' in b and any(p['path'] and len(p['path'])>=1 for p in projects[:4]):return b['play_road_building'][0]
        # Prefer profitable complete projects, not just the first affordable action.
        for p in projects:
            if not p['ready']:continue
            if s.vp<9 and p['type']=='buy_development_card' and p.get('net_gain_after_cost',0)<-0.10:
                self.ctx.log('economic_saving',declined=p['id'],net_gain=p['net_gain_after_cost'],preferred=projects[0]['id'],hand=s.hand)
                continue
            trades=s.trades(p['cost'])
            if trades:
                a=s.first_trade(trades)
                if a:
                    self.commit(s,p)
                    return a
            a=s.project_action(p)
            if a:
                self.commit(s,p)
                return a
        # Pay for a route increment only when saving for a nearby settlement is sensible.
        if projects:
            p=projects[0]
            if p['path'] and p['eta']<=3:
                a=s.project_action(p)
                if a:return a
        if 'end_turn' in b:return b['end_turn'][0]
        if 'build_road' in b:return self.best_road(s,projects)
        return s.legal[0]

    def commit(self,s,p):
        if p['type']!='award':
            self.active_project=p['id'];self.active_turn=s.o.get('turn_number')

    def best_road(self,s,projects):
        for p in projects:
            if p.get('path'):
                a=s.find('build_road',p['path'][0])
                if a:return a
        return self.initial_road(s)

    def run_episode(self,game):
        ctx=self.ctx; obs=ctx.observe(); self.wall_deadline=time.monotonic()+ctx.budget.remaining().get("wall_s",1800)
        while not ctx.done:
            if min(ctx.budget.remaining().get("wall_s",0), self.wall_deadline-time.monotonic())<max(45.0,2*self.max_step_s+5):break
            if not isinstance(obs,dict):obs=ctx.observe()
            if obs.get('terminal'):break
            s=State(obs)
            s.road_awards_enabled=self.road_awards_enabled
            if not s.legal:break
            turn=obs.get('turn_number',0)
            if self.last_revision!=obs.get('revision'):
                self.bad.clear();self.last_revision=obs.get('revision')
            if s.vp!=self.last_score:self.last_score=s.vp;self.last_score_turn=turn
            projects=s.projects(self.mode,self.preferred) if not s.setup else []
            if self.investment is not None and not s.setup:
                record=self.investment
                if s.vp>record['vp']:
                    ctx.log('investment_conversion',investment=record,next_vp=s.vp,turn=turn)
                    self.investment=None
                elif turn>=record['review_turn'] and 'end_turn' in s.bytype:
                    self.failure='Production investment did not convert into another point within six own rounds: '+json.dumps(record)+'; progress '+json.dumps(ctx.progress())
                    ctx.log('economic_plan_failure',reason=self.failure)
                    self.investment=None
            a=self.poll(s,projects); reason=None
            if self.failure:reason='Execution or plan failure: '+self.failure
            elif s.setup and 'build_settlement' in s.bytype:reason='Choose complementary opening settlement'
            elif not s.setup and 'end_turn' in s.bytype:
                threat=max([0]+[x.get('public_vp',0) for x in obs.get('opponents',[])])
                if self.last_vp<2:reason='Opening complete: choose economic strategy'
                elif self.preferred and self.preferred not in {p['id'] for p in projects}:
                    self.preferred=None;reason='Preferred project completed or became unavailable; update portfolio'
                elif turn-self.last_call_turn>=20 and (s.vp>=self.last_vp+2 or threat>=8>self.last_threat):reason='Victory race changed'
                elif turn-self.last_call_turn>=28 and turn-self.last_score_turn>=28:
                    self.failure='Seven own rounds without score growth; reconsider resource bottleneck; progress '+json.dumps(ctx.progress())
                    reason=self.failure
            if reason and (self.pending is None or self.failure):
                calls_before=self.calls
                a=self.ask(s,projects,reason,s.setup)
                if self.calls>calls_before:self.failure=None
                projects=s.projects(self.mode,self.preferred) if not s.setup else []
            if ctx.done:break
            if a is None:
                projects=s.projects(self.mode,self.preferred) if not s.setup else []
                a=self.choose(s,projects)
            if key(a) in self.bad:
                alternatives=[x for x in s.legal if key(x) not in self.bad]
                if not alternatives:break
                a=alternatives[0]
            if not s.setup and a['type'] in ('build_city','build_settlement','buy_development_card','maritime_trade'):
                ctx.log('economic_choice',action=a,hand=s.hand,production=s.production,recipes=getattr(s,'victory_recipes',[])[:3],projects=projects[:5])
            before=ctx.progress(); revision=obs.get('revision'); oldobs=obs; step_start=time.monotonic()
            try:
                obs=ctx.act(a['id'])
                if obs is None:obs=ctx.observe()
            except BudgetExceeded:
                raise
            except Exception as exc:
                if ctx.done:break
                self.failure='Action raised '+str(exc)[:160];self.bad.add(key(a));self.rejects+=1
                obs=ctx.observe()
                if self.rejects>=6:break
                continue
            self.max_step_s=max(self.max_step_s,time.monotonic()-step_start)
            after=ctx.progress()
            rejected=(after.get('illegal_actions',0)>before.get('illegal_actions',0) or
                      (revision is not None and isinstance(obs,dict) and obs.get('revision')==revision) or
                      after.get('since_obs_change',0)>=2)
            if rejected:
                self.bad.add(key(a));self.rejects+=1;self.active_project=None
                self.failure='Rejected or unchanged action '+json.dumps(a)+'; progress '+json.dumps(after)
            else:
                self.rejects=0
                self.verify(oldobs,obs,a,after)
            if self.rejects>=6:break

    def verify(self,old,new,a,progress):
        if not isinstance(new,dict) or new.get('terminal'):return
        typ=a['type']; p=a.get('parameters'); color=old.get('self',{}).get('id')
        if typ in ('build_city','build_settlement','buy_development_card'):self.active_project=None
        failure=None
        if typ in ('build_settlement','build_city'):
            vertex=next((v for v in new.get('board',{}).get('vertices',[]) if v['id']==p),{})
            expected='CITY' if typ=='build_city' else 'SETTLEMENT'
            if vertex.get('owner')!=color or vertex.get('building')!=expected:failure='Building postcondition failed'
            elif 'initial' not in str(old.get('phase','')):
                turn=new.get('turn_number',0)
                self.investment={'action':a['type'],'vertex':p,'turn':turn,'vp':new.get('self',{}).get('true_vp',0),'review_turn':turn+24}
                self.ctx.log('investment_started',investment=self.investment)
        elif typ=='build_road':
            edge=next((e for e in new.get('board',{}).get('edges',[]) if sorted(e['id'])==sorted(p)),{})
            if edge.get('owner')!=color:failure='Road postcondition failed'
            elif self.road_awards_enabled:
                before=old.get('awards',{});after=new.get('awards',{})
                needed=max(5,before.get('longest_road_length',0)+1)
                if (before.get('longest_road_owner')!=color and
                    State(new).road_length()>=needed and after.get('longest_road_owner')!=color):
                    self.road_awards_enabled=False
                    failure='Road-award prediction contradicted engine; disabled award-only projects'

        elif typ=='maritime_trade' and isinstance(p,list):
            oldhand=old.get('self',{}).get('resources',{}); newhand=new.get('self',{}).get('resources',{})
            expected=dict(oldhand)
            for r in p[:-1]:
                if r in R:expected[r]=expected.get(r,0)-1
            if p[-1] in R:expected[p[-1]]=expected.get(p[-1],0)+1
            if expected!=newhand:failure='Trade cost/receipt differed from the project model'
        if failure:
            self.failure=failure+'; '+json.dumps(progress); self.preferred=None;self.active_project=None
            self.ctx.log('project_postcondition_failure',reason=self.failure)


def build(ctx):
    return Harness(ctx)
