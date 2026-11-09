"""Staged expansion financing with live milestone contracts and Astra recovery.
Public observations only; current legal IDs and postconditions govern execution.
"""
from __future__ import annotations
import json
import heapq
import re
import time
from metaharness.core.contract import BudgetExceeded


class PlanningStopped(Exception):
    """Cooperative stop before a costly game call or local search overruns."""


R = ('WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE')
COST = {'build_road': (1,1,0,0,0), 'build_settlement': (1,1,1,1,0),
        'build_city': (0,0,0,2,3), 'buy_development_card': (0,0,1,1,1)}
PIPS = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}

def key(a):
    return (a['type'], json.dumps(a.get('parameters'), sort_keys=True))

def parse(text):
    if not isinstance(text,str): return {}
    for i,c in enumerate(text):
        if c == '{':
            try:
                x,_ = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(x,dict): return x
            except (ValueError,TypeError): pass
    return {}

class State:
    def __init__(self,o,check=None):
        self.check=check or (lambda: None)
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
            self.check()
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
            self.check()
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
            self.check()
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
                self.check()
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

class Harness:
    def __init__(self,ctx):
        self.ctx=ctx; self.mode='balanced'; self.preferred=None; self.memo=''
        self.calls=0; self.last_call_turn=-999; self.last_vp=-1; self.last_threat=0
        self.failure=None; self.rejects=0; self.last_score_turn=0; self.last_score=-1
        self.bad=set(); self.last_revision=None
        self.pending=None; self.pending_revision=None
        self.active_project=None; self.active_turn=None; self.road_awards_enabled=True

    def accept(self,s,projects,result,allow_action):
        reply=getattr(result,"text","")
        if not isinstance(reply,str): reply=""
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
        # Pay for a route increment whenever it is a verified first step toward
        # an open productive site.  The old executor required ETA<=3 here;
        # that threshold made it end turns forever on low-production starts,
        # even though a legal road was the only way to change the economy.
        if projects:
            p=projects[0]
            if p['path']:
                a=s.project_action(p)
                if a:return a
        # If the portfolio's route was invalidated by an opponent, choose the
        # legal frontier edge with the best newly reachable site.  This is a
        # bounded local repair, not a blind road race.
        if 'build_road' in b:
            a=self.frontier_road(s)
            if a:return a
        if 'end_turn' in b:return b['end_turn'][0]
        if 'build_road' in b:return self.best_road(s,projects)
        return s.legal[0]

    def frontier_road(self,s):
        base=s.routes
        def score(a):
            try: ex=s.find_routes([a.get('parameters')])
            except Exception:return -1e9
            gain=[]
            for n,path in ex.items():
                if n not in base or len(path)<len(base.get(n,[])):
                    gain.append(s.pip_value(n)/(1+len(path)))
            # Prefer roads that preserve a settlement endpoint and diversity.
            return max(gain or [0])-0.03*len(s.own_edges)
        return max(s.bytype.get('build_road',[]),key=score,default=None)

    def commit(self,s,p):
        if p['type']!='award':
            self.active_project=p['id'];self.active_turn=s.o.get('turn_number')

    def best_road(self,s,projects):
        for p in projects:
            if p.get('path'):
                a=s.find('build_road',p['path'][0])
                if a:return a
        return self.initial_road(s)

    def verify(self,old,new,a,progress):
        if not isinstance(new,dict) or new.get('terminal'):return
        typ=a['type']; p=a.get('parameters'); color=old.get('self',{}).get('id')
        failure=None
        if typ in ('build_settlement','build_city'):
            vertex=next((v for v in new.get('board',{}).get('vertices',[]) if v['id']==p),{})
            expected='CITY' if typ=='build_city' else 'SETTLEMENT'
            if vertex.get('owner')!=color or vertex.get('building')!=expected:failure='Building postcondition failed'
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




class StagedHarness(Harness):
    """One live construction contract; each milestone has separate financing."""
    def __init__(self,ctx):
        super().__init__(ctx)
        self.contract=None; self.cooldown={}; self.no_expansion_until=-1
        self.retired=[]; self.pending_reason=None; self.failure_serial=0
        self.real_deadline=None; self.max_game_call_s=0.0
        self.last_failure_asked=None; self.last_stage_report=None
        # A bounded public-race interrupt.  It is deliberately separate from
        # the construction contract so a threat cannot silently rewrite it.
        self.threat_until=-1
        self.threat_color=None
        self.threat_epoch=0
        # Conversion urgency is an evidence gate, not a timer-only strategy.
        # It fires once after a long run of accepted actions without any VP
        # change, then cools down so it cannot consume the whole model budget.
        self.conversion_until=-1
        self.last_conversion_signal=-999

    def remaining_s(self):
        r=float(self.ctx.budget.remaining().get('wall_s',0))
        return min(r,self.real_deadline-time.monotonic()) if self.real_deadline else r

    def check_work(self):
        if self.ctx.done or self.remaining_s()<=max(45.0,3*self.max_game_call_s+5):
            raise PlanningStopped()

    def game_call(self,fn,*args):
        self.check_work();t=time.monotonic()
        try:return fn(*args)
        finally:self.max_game_call_s=max(self.max_game_call_s,time.monotonic()-t)

    def signal(self,message):
        self.failure_serial+=1
        self.failure=str(message)+'; progress='+json.dumps(self.ctx.progress(),separators=(',',':'))
        self.ctx.log('staged_failure',reason=self.failure,serial=self.failure_serial)

    def abandon(self,s,reason,failed=True):
        if self.contract:
            self.ctx.log('expansion_abandoned',contract=self.contract,reason=reason)
            self.cooldown[self.contract['id']]=s.o.get('turn_number',0)+16
            self.contract=None
        self.active_project=None
        if failed:self.signal(reason)

    def contract_project(self,projects):
        return next((p for p in projects if self.contract and p['id']==self.contract['id']),None)

    def monitor(self,s,projects):
        if not self.contract:return
        c=self.contract;turn=s.o.get('turn_number',0);p=self.contract_project(projects)
        if c['target'] in s.mine:
            self.ctx.log('expansion_completed',contract=c,turn=turn,elapsed_turns=turn-c['start'])
            self.contract=None;self.active_project=None;return
        if p is None:
            self.abandon(s,'Construction target lost or no legal public route remains');return
        if len(p['path'])>c['roads_left']:
            self.abandon(s,'Construction route lengthened from '+str(c['roads_left'])+' to '+str(len(p['path'])));return
        if len(p['path'])<c['roads_left']:
            c['roads_left']=len(p['path']);c['milestone_deadline']=turn+24
            c['paid_stages']+=1
            self.ctx.log('expansion_milestone',contract=dict(c),remaining_cost=p['cost'])
        if turn>=c['deadline'] or turn>=c['milestone_deadline']:
            self.abandon(s,'Construction deadline missed: '+json.dumps(c)+' remaining hand '+str(s.hand));return
        # A public network can reach a site without having the cards to settle.
        # Distances are a threat flag, never an assertion of hidden affordability.
        c['rivals']=self.rival_distances(s,p['target'])
        c['remaining_cost']=p['cost']
        c['deficits']=[max(0,p['cost'][i]-s.hand[i]) for i in range(5)]

    def rival_distances(self,s,target):
        result={}
        forbidden={target,*s.v[target].get('neighbors',[])}
        for op in s.o.get('opponents',[]):
            color=op['id'];sources={n for n,v in s.v.items() if v.get('owner')==color}
            for edge,owner in s.edges.items():
                if owner==color:sources.update(n for n in edge if s.v[n].get('owner') in (None,color))
            dist={n:0 for n in sources};q=[(0,n) for n in sources];heapq.heapify(q)
            while q:
                d,n=heapq.heappop(q)
                if d!=dist[n] or d>3:continue
                if s.v[n].get('owner') not in (None,color):continue
                for z in s.v[n].get('neighbors',[]):
                    own=s.edges.get(tuple(sorted((n,z))))
                    if own not in (None,color):continue
                    nd=d+(own!=color)
                    if nd<dist.get(z,99):dist[z]=nd;heapq.heappush(q,(nd,z))
            result[color]=min([dist.get(n,99) for n in forbidden if s.open_site(n)] or [99])
        return result

    def acquire(self,s,p,source):
        turn=s.o.get('turn_number',0)
        if self.contract and self.contract['id']==p['id']:return
        self.contract={'id':p['id'],'target':p['target'],'start':turn,
                       'roads_left':len(p['path']),'paid_stages':0,
                       'deadline':turn+32+8*len(p['path']),
                       'milestone_deadline':turn+24,'source':source,
                       'rivals':self.rival_distances(s,p['target']),
                       'remaining_cost':p['cost'],
                       'deficits':[max(0,p['cost'][i]-s.hand[i]) for i in range(5)]}
        self.active_project=None
        self.ctx.log('expansion_acquired',contract=dict(self.contract),full_cost=p['cost'],hand=s.hand)

    def stage_cost(self,p):
        return COST['build_road'] if p['path'] else COST['build_settlement']

    def stage_funding(self,s,p):
        """Enumerate integer milestone trades, retaining the rest of the project.
        No back-and-forth trades, and every returned first step is live legal.
        """
        stage=self.stage_cost(p);results=[]
        def rec(h,b,path):
            self.check_work()
            deficit=[i for i in range(5) if h[i]<stage[i]]
            if not deficit:
                remaining=[max(0,p['cost'][i]-stage[i]) for i in range(5)]
                after=[h[i]-stage[i] for i in range(5)]
                # Preserve the most scarce future target cards, then total cards.
                loss=sum(max(0,remaining[i]-after[i])/(.25+s.production[i]) for i in range(5))
                results.append((loss,len(path),-sum(after),path));return
            if len(path)>=4:return
            j=min(deficit,key=lambda i:b[i])
            if b[j]<=0:return
            for i in range(5):
                if h[i]-stage[i]<s.rate[i]:continue
                hh=list(h);bb=list(b);hh[i]-=s.rate[i];hh[j]+=1;bb[i]+=s.rate[i];bb[j]-=1
                rec(hh,bb,path+[(i,j,s.rate[i])])
        rec(list(s.hand),list(s.bank),[])
        if not results:return None
        return min(results,key=lambda x:x[:3])[3]

    def partial_trade(self,s,p):
        """Bank surplus toward a milestone only if it reduces exposed inventory.
        Never export a card that is still needed by the complete contract.
        """
        if sum(s.hand)<=7:return None
        stage=self.stage_cost(p);options=[]
        for a in s.bytype.get('maritime_trade',[]):
            pp=a.get('parameters',[])
            if not isinstance(pp,list) or not pp or pp[-1] not in R:continue
            give=[x for x in pp[:-1] if x in R]
            if not give or len(set(give))!=1:continue
            i=R.index(give[0]);j=R.index(pp[-1])
            if s.hand[i]-len(give)<p['cost'][i] or s.hand[j]>=stage[j]:continue
            options.append((1/(.25+s.production[j]),-s.rate[i],a))
        return max(options,key=lambda x:x[:2])[2] if options else None

    def select_contract(self,s,projects):
        if self.contract or s.o.get('turn_number',0)<self.no_expansion_until:return
        turn=s.o.get('turn_number',0);options=[]
        for p in projects:
            if p['type']!='build_settlement' or len(p['path'])>2:continue
            if self.cooldown.get(p['id'],-1)>turn:continue
            if sum(s.site[p['target']])<3 and not s.v[p['target']].get('port'):continue
            funding=self.stage_funding(s,p)
            if funding is None and self.partial_trade(s,p) is None:continue
            rivals=self.rival_distances(s,p['target']);nearest=min(rivals.values(),default=99)
            # If an enemy already reaches the site, prefer a complete build now.
            if nearest==0 and p['path'] and not p['ready']:continue
            value=p['score']*(1.25 if nearest<=2 else 1.0)
            if p['id']==self.preferred:value*=1.25
            options.append((value,p))
        if not options:return
        value,p=max(options,key=lambda x:x[0])
        funded_builds=[x for x in projects if x['ready'] and x['type'] in ('build_city','build_settlement','award')]
        if funded_builds and max(x['score'] for x in funded_builds)>value*1.3:return
        self.acquire(s,p,'local staged financing')

    def choose(self,s,projects):
        b=s.bytype
        if s.setup or 'roll' in b or 'move_robber' in b:
            return super().choose(s,projects)
        # When a rival is visibly at the finish line, opportunity cost changes:
        # release a speculative expansion and convert any funded point now.
        # Every candidate still comes from the current legal menu.
        if s.o.get('turn_number',0) < self.threat_until:
            for p in projects:
                if not p.get('ready') or p.get('type') not in ('build_city','build_settlement'):
                    continue
                ts=s.trades(p['cost'])
                a=s.first_trade(ts) if ts else s.project_action(p)
                if a:return a
            # A direct legal build can appear before the project compiler has
            # refreshed its portfolio after a bot action.
            for typ in ('build_city','build_settlement'):
                if b.get(typ):
                    return max(b[typ],key=lambda a:s.pip_value(a.get('parameters')))
            # At nine VP, a funded development card is a legitimate immediate
            # finish chance; do not buy a card that still needs speculative
            # resources during the interrupt.
            if s.vp>=9:
                for p in projects:
                    if p.get('ready') and p.get('type')=='buy_development_card':
                        ts=s.trades(p['cost'])
                        a=s.first_trade(ts) if ts else s.project_action(p)
                        if a:return a
            # Tactical cards (especially Monopoly/Year of Plenty) are handled
            # by the parent's legal, resource-aware policy.
            tactical=super().choose(s,projects)
            if tactical and tactical.get('type') not in ('end_turn','roll'):
                return tactical
        # Once stagnation evidence has been sent to Astra, keep the executor
        # in conversion mode for a short window.  This closes the failure mode
        # where a valid construction contract repeatedly ends turns while a
        # funded point or exact maritime step is available.
        if s.o.get('turn_number',0) < self.conversion_until:
            for p in projects:
                if p.get('ready') and p.get('type') in ('build_city','build_settlement','award'):
                    ts=s.trades(p['cost']); a=s.first_trade(ts) if ts else s.project_action(p)
                    if a:return a
            for typ in ('build_city','build_settlement'):
                if b.get(typ):return max(b[typ],key=lambda a:s.pip_value(a.get('parameters')))
            # A legal tactical card can create the missing conversion; defer
            # only rolls/end-turns, which are reconsidered next observation.
            tactical=super().choose(s,projects)
            if tactical and tactical.get('type') not in ('end_turn','roll'):
                return tactical
        p=self.contract_project(projects)
        if 'discard_resource' in b:
            goal=p['cost'] if p else (projects[0]['cost'] if projects else COST['build_city'])
            def discard_value(a):
                i=R.index(a['parameters']);surplus=max(0,s.hand[i]-goal[i])
                return (surplus,-goal[i]/(.3+s.production[i]),s.hand[i])
            return max(b['discard_resource'],key=discard_value)
        if set(b)=={'build_road'}:
            if p and p['path']:
                a=s.find('build_road',p['path'][0])
                if a:return a
            return super().choose(s,projects)
        # A funded scoring finish supersedes an unfinished construction job.
        for win in projects:
            if not win['ready']:continue
            winning=((s.vp>=9 and win['type'] in ('build_city','build_settlement')) or
                     (s.vp>=8 and win['type']=='award'))
            if winning:
                ts=s.trades(win['cost'])
                a=s.first_trade(ts) if ts else s.project_action(win)
                if a:return a
        # Use the parent's tactical card policy and immediate-win safeguards.
        ordered=([p]+[x for x in projects if x['id']!=p['id']]) if p else projects
        base=super().choose(s,ordered)
        if s.vp>=9 or base['type'].startswith('play_'):return base
        if not self.contract:
            self.select_contract(s,projects);p=self.contract_project(projects)
        if not p:return base
        funded_city=next((x for x in projects if x['type']=='build_city' and x['ready']),None)
        # Productive upgrades are allowed to accelerate a long construction job.
        if funded_city and p['path'] and funded_city['score']>p['score']*1.3:
            ts=s.trades(funded_city['cost'])
            a=s.first_trade(ts) if ts else s.project_action(funded_city)
            if a:return a
        trades=self.stage_funding(s,p)
        if trades is not None:
            a=s.first_trade(trades) if trades else s.project_action(p)
            if a:
                if trades:self.ctx.log('stage_funding',target=p['id'],stage_cost=self.stage_cost(p),
                                       full_cost=p['cost'],full_ready=p['ready'],action=a)
                return a
        a=self.partial_trade(s,p)
        if a:
            self.ctx.log('stage_partial_trade',target=p['id'],action=a,hand=s.hand)
            return a
        # Maintain the construction milestone across turns. Cheap dev purchases
        # are allowed only if they do not increase the current milestone deficit.
        if base['type']=='buy_development_card':
            cost=COST['buy_development_card'];stage=self.stage_cost(p)
            old=sum(max(0,stage[i]-s.hand[i])/(.25+s.production[i]) for i in range(5))
            new=sum(max(0,stage[i]-s.hand[i]+cost[i])/(.25+s.production[i]) for i in range(5))
            if new<=old+1e-9:return base
        if base['type']=='build_city' and p['path']:return base
        # Avoid unrelated roads/trades consuming the held milestone resources.
        return s.find('end_turn') or base

    def ask(self,s,projects,reason,opening=False):
        self.retired=[f for f in self.retired if not f.done()]
        if self.pending is not None:
            if self.failure and self.pending_reason!=self.failure_serial and not self.retired:
                self.retired.append(self.pending);self.pending=None
                self.ctx.log('superseded_planner',reason=self.failure)
            else:return self.poll(s,projects)
        if self.calls>=14 or self.ctx.budget.remaining().get('system2_calls',0)<=1 or self.remaining_s()<370:return None
        choices=[]
        if opening:
            for a in s.bytype.get('build_settlement',[]):
                n=a['parameters'];choices.append(dict(a,pips=s.site[n],port=s.v[n].get('port'),value=round(s.opening_value(n),2)))
            choices.sort(key=lambda x:x['value'],reverse=True)
        payload={'reason':reason,'failure':self.failure,'progress':self.ctx.progress(),
                 'turn':s.o.get('turn_number'),'phase':s.o.get('phase'),'self':s.me,
                 'opponents':s.o.get('opponents'),'awards':s.o.get('awards'),
                 'resource_order':R,'production_per_round':[round(x,2) for x in s.production],
                 'our_sites':[{'vertex':n,'pips':s.site[n],'building':s.v[n].get('building')} for n in s.mine],
                 'contract':self.contract,'prior_plan':self.memo,'mode':self.mode,
                 'public_threat':{'until':self.threat_until,'color':self.threat_color,
                                  'epoch':self.threat_epoch},
                 'projects':projects[:12],'opening_choices':choices,
                 'expansion_options':[dict(p,rival_road_distance=self.rival_distances(s,p['target']),
                                          next_milestone_cost=self.stage_cost(p))
                                      for p in projects if p['type']=='build_settlement' and len(p['path'])<=2][:10],
                 'legal_actions':[] if opening else s.legal,
                 'recent_events':s.o.get('events',[])[-10:]}
        prompt=('You play Catan against three strong bots. Reach 10 true VP. Choose a strategy and manage ONE construction contract. '
          'The executor can fund an expansion in installments: trade for a road, build it, then save/trade for its settlement. '
          'Avoid stranded roads; once a site is reached finish it before rivals. A road is not a VP. '
          'Cities can accelerate construction; Army and hidden VP remain alternative win paths. '
          'Production and ETA are approximate, not guaranteed. Rival distance is public connectivity, NOT known affordability. '
          'The contract expires and reopens planning on a lost target or stalled milestone. '
          'Reply ONLY compact JSON {"action":null,"mode":"balanced","project":null,"expansion":"auto","plan":"under 35 words"}. '
          'mode is balanced/cities/expand/army. project is an offered id or null. expansion is an offered settle:id to commit, '
          '"auto" to retain/local select, or "stop" to release and pause expansion for six rounds. '
          'Use a current action ID for setup or an immediate tactical recovery. On failure revise rather than repeating the failed target. '
          'Code verifies each action against fresh legality and effects.\n'+json.dumps(payload,separators=(',',':')))
        self.calls+=1;self.last_call_turn=s.o.get('turn_number',0);self.last_vp=s.vp
        self.last_threat=max([0]+[x.get('public_vp',0) for x in s.o.get('opponents',[])])
        self.pending_revision=s.o.get('revision');self.pending_reason=self.failure_serial
        self.pending=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,timeout_s=300.0))
        self.last_failure_asked=self.failure_serial;self.failure=None
        try:
            result=self.pending.result(timeout=min(210.0,self.remaining_s()-50))
            self.pending=None
            return self.accept(s,projects,result,True)
        except BudgetExceeded:raise
        except TimeoutError:
            self.ctx.log('planner_wait_limit',seconds=210);return None
        except Exception as exc:
            self.pending=None;self.ctx.log('planner_unavailable',error=str(exc)[:200]);return None

    def accept(self,s,projects,result,allow_action):
        reply=getattr(result,'text','');ans=parse(reply)
        exp=ans.get('expansion')
        if exp=='stop':
            self.abandon(s,'Astra selected another victory path',False)
            self.no_expansion_until=s.o.get('turn_number',0)+24
        elif isinstance(exp,str):
            p=next((p for p in projects if p['id']==exp and p['type']=='build_settlement'),None)
            if p is not None and len(p['path'])<=3:self.acquire(s,p,'Astra')
        return super().accept(s,projects,result,allow_action)

    def poll(self,s,projects):
        if self.pending is None or not self.pending.done():return None
        future=self.pending;self.pending=None
        try:
            result=future.result()
            # An answer predating new failure evidence cannot restore a bad plan.
            if self.pending_reason!=self.failure_serial:return None
            return self.accept(s,projects,result,s.o.get('revision')==self.pending_revision)
        except BudgetExceeded:raise
        except Exception as exc:
            self.ctx.log('planner_unavailable',error=str(exc)[:200]);return None

    def verify(self,old,new,a,progress):
        old_failure=self.failure
        super().verify(old,new,a,progress)
        # Check financing against actual cards as well as board ownership.
        # Initial and free-road placements have no ordinary resource cost.
        if isinstance(new,dict) and not new.get('terminal'):
            kinds={x['type'] for x in old.get('legal_actions',[])}
            typ=a['type']
            if typ in COST and 'end_turn' in kinds:
                before=old.get('self',{}).get('resources',{})
                after=new.get('self',{}).get('resources',{})
                expected={r:before.get(r,0)-COST[typ][i] for i,r in enumerate(R)}
                if any(after.get(r,0)!=expected[r] for r in R):
                    self.failure='Paid milestone resource cost contradicted observation; '+json.dumps(progress)
        if self.failure and self.failure!=old_failure:
            self.contract=None;self.failure_serial+=1
            self.ctx.log('stage_postcondition_failure',reason=self.failure)

    def run_episode(self,game):
        self.real_deadline=time.monotonic()+max(0,float(self.ctx.budget.remaining().get('wall_s',0)))
        try:self.loop(game)
        except (BudgetExceeded,PlanningStopped):return

    def loop(self,game):
        ctx=self.ctx;obs=self.game_call(ctx.observe)
        while not ctx.done:
            self.check_work()
            if not isinstance(obs,dict):obs=self.game_call(ctx.observe)
            if obs.get('terminal'):return
            s=State(obs,check=self.check_work);s.road_awards_enabled=self.road_awards_enabled
            if not s.legal:return
            turn=obs.get('turn_number',0)
            if self.last_revision!=obs.get('revision'):
                self.bad.clear();self.last_revision=obs.get('revision')
            if s.vp!=self.last_score:self.last_score=s.vp;self.last_score_turn=turn
            projects=s.projects(self.mode,self.preferred) if not s.setup else []
            if not s.setup:self.monitor(s,projects)
            # Detect accepted-but-unproductive play from the authoritative
            # progress counters.  A contract can remain legal while its
            # opportunity cost grows; hand the concrete evidence to Astra and
            # temporarily require a point conversion or a funded step.
            prog=ctx.progress()
            stagnant=int(prog.get('since_score_change',0) or 0)
            if (not s.setup and stagnant>=48 and turn-self.last_conversion_signal>=24
                    and turn>=self.last_score_turn+12):
                if self.contract:
                    self.abandon(s,'VP stagnation after accepted actions; release construction contract')
                self.conversion_until=turn+16
                self.last_conversion_signal=turn
                self.signal('VP stagnation requires conversion; '+json.dumps(prog,separators=(',',':')))
                self.ctx.log('conversion_urgency',turn=turn,progress=prog,until=self.conversion_until)
            # Public threat bounds are intentionally conservative: a rival at
            # 9 public VP can finish with a hidden VP, while 8 public VP is
            # already urgent when our own true VP is below 9.  The interrupt
            # lasts six controlled rounds (24 global turn numbers), then the
            # normal contract policy is allowed to resume.
            opponents=s.o.get('opponents',[])
            threat=max([0]+[int(p.get('public_vp',0) or 0) for p in opponents])
            urgent=(threat>=9 or (threat>=8 and s.vp<9))
            turn_now=s.o.get('turn_number',0)
            if urgent and turn_now>=self.threat_until:
                self.threat_until=turn_now+24
                self.threat_color=next((p.get('id') for p in opponents
                                        if int(p.get('public_vp',0) or 0)==threat),None)
                self.threat_epoch+=1
                if self.contract:
                    self.abandon(s,'Public opponent threat '+str(threat)+' VP; release expansion',False)
                self.signal('Public opponent threat '+str(threat)+' VP; prioritize immediate conversion')
                self.ctx.log('public_threat_interrupt',threat=threat,color=self.threat_color,
                             until=self.threat_until,epoch=self.threat_epoch)
            a=self.poll(s,projects);reason=None
            if self.failure:reason=self.failure
            elif s.setup and 'build_settlement' in s.bytype:reason='Choose complementary opening settlement'
            elif 'end_turn' in s.bytype:
                if self.last_vp<2:reason='Opening complete; choose first economic investment and expansion target'
                elif turn-self.last_call_turn>=20 and (s.vp>=self.last_vp+2 or threat>=8>self.last_threat):reason='Victory race changed'
                elif turn-self.last_call_turn>=24 and turn-self.last_score_turn>=24:reason='Six rounds without VP growth: revise economic plan'
            if reason and (self.pending is None or self.failure):
                a=self.ask(s,projects,reason,s.setup)
                projects=s.projects(self.mode,self.preferred) if not s.setup else []
            if a is None:a=self.choose(s,projects)
            if key(a) in self.bad:
                alt=[x for x in s.legal if key(x) not in self.bad]
                if not alt:return
                a=next((x for x in alt if x['type']=='end_turn'),alt[0])
            before=ctx.progress();revision=obs.get('revision');old=obs
            try:
                obs=self.game_call(ctx.act,a['id'])
                if obs is None:obs=self.game_call(ctx.observe)
            except (BudgetExceeded,PlanningStopped):raise
            except Exception as exc:
                if ctx.done:return
                self.signal('Action raised '+str(exc)[:150]);self.contract=None
                self.bad.add(key(a));self.rejects+=1;obs=self.game_call(ctx.observe)
                if self.rejects>=6:return
                continue
            after=ctx.progress()
            rejected=(after.get('illegal_actions',0)>before.get('illegal_actions',0) or
                      (revision is not None and isinstance(obs,dict) and obs.get('revision')==revision) or
                      after.get('since_obs_change',0)>=2)
            if rejected:
                self.bad.add(key(a));self.rejects+=1;self.contract=None;self.active_project=None
                self.signal('Rejected/unchanged action '+json.dumps(a))
            else:
                self.rejects=0;self.verify(old,obs,a,after)
            if self.rejects>=6:return


def build(ctx):
    return StagedHarness(ctx)
