"""Economic projects, with Astra as a sparse strategic portfolio manager.
All game knowledge comes from public observations; execution uses live legal IDs.
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
        self.real_deadline=None; self.max_game_call_s=0.0

    def remaining_s(self):
        remaining=float(self.ctx.budget.remaining().get('wall_s',0))
        if self.real_deadline is not None:
            remaining=min(remaining,self.real_deadline-time.monotonic())
        return remaining

    def check_work(self):
        # ctx's virtual clock need not charge CPU work; the evaluator's real
        # watchdog still does. Never start a synchronous bot advance at the
        # boundary. It cannot safely be cancelled from a background thread.
        reserve=max(45.0,3.0*self.max_game_call_s+5.0)
        if self.ctx.done or self.remaining_s()<=reserve:
            raise PlanningStopped()

    def game_call(self,fn,*args):
        self.check_work()
        started=time.monotonic()
        try:
            return fn(*args)
        finally:
            self.max_game_call_s=max(self.max_game_call_s,time.monotonic()-started)


    def ask(self,s,projects,reason,opening=False):
        ctx=self.ctx; budget=ctx.budget.remaining()
        if self.pending is not None:
            return self.poll(s,projects)
        if self.calls>=12 or budget.get('system2_calls',0)<=1 or self.remaining_s()<380: return None
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
                 'recent_events':s.o.get('events',[])[-12:]}
        if not opening:
            payload['legal_actions']=s.legal
            payload['expansion_sites']=[{'vertex':n,'pips':s.site[n],'path':path,'port':s.v[n].get('port')} for n,path in sorted(s.routes.items(),key=lambda x:s.pip_value(x[0]),reverse=True)[:10]]
        prompt=("You manage a Catan economic planner against three strong AlphaBeta bots. Reach 10 true VP first. "
          "Choose a coherent route to victory from code-computed projects. Roads without useful endpoints are wasteful; "
          "ore/wheat city production compounds, development cards give army and hidden VP. Army incumbent wins ties. "
          "Production/ETA are approximate expected values, not guaranteed rolls. Hidden opponent cards are unknown. "
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
            self.check_work()
            result=self.pending.result(timeout=min(210.0,self.remaining_s()-45.0))
            self.pending=None
            return self.accept(s,projects,result,True)
        except (BudgetExceeded,PlanningStopped):
            raise
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
        except (BudgetExceeded,PlanningStopped):
            raise
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
        self.real_deadline=time.monotonic()+max(0.0,float(self.ctx.budget.remaining().get('wall_s',0)))
        try:
            self._run_episode(game)
        except (BudgetExceeded,PlanningStopped):
            return

    def _run_episode(self,game):
        ctx=self.ctx; obs=self.game_call(ctx.observe)
        while not ctx.done:
            self.check_work()
            if not isinstance(obs,dict):obs=self.game_call(ctx.observe)
            if obs.get('terminal'):break
            s=DiceState(obs,check=self.check_work)
            s.road_awards_enabled=self.road_awards_enabled
            if not s.legal:break
            turn=obs.get('turn_number',0)
            if self.last_revision!=obs.get('revision'):
                self.bad.clear();self.last_revision=obs.get('revision')
            if s.vp!=self.last_score:self.last_score=s.vp;self.last_score_turn=turn
            projects=s.projects(self.mode,self.preferred) if not s.setup else []
            a=self.poll(s,projects); reason=None
            if self.failure:reason='Execution or plan failure: '+self.failure
            elif s.setup and 'build_settlement' in s.bytype:reason='Choose complementary opening settlement'
            elif not s.setup and 'end_turn' in s.bytype:
                threat=max([0]+[x.get('public_vp',0) for x in obs.get('opponents',[])])
                if self.last_vp<2:reason='Opening complete: choose economic strategy'
                elif self.preferred and self.preferred not in {p['id'] for p in projects}:
                    self.preferred=None;reason='Preferred project completed or became unavailable; update portfolio'
                elif turn-self.last_call_turn>=20 and (s.vp>=self.last_vp+2 or threat>=8>self.last_threat):reason='Victory race changed'
                elif turn-self.last_call_turn>=28 and turn-self.last_score_turn>=28:reason='Seven own rounds without score growth; reconsider resource bottleneck'
            if reason and self.pending is None:
                a=self.ask(s,projects,reason,s.setup)
                self.failure=None
                projects=s.projects(self.mode,self.preferred) if not s.setup else []
            if ctx.done:break
            if a is None:
                projects=s.projects(self.mode,self.preferred) if not s.setup else []
                a=self.choose(s,projects)
            if key(a) in self.bad:
                alternatives=[x for x in s.legal if key(x) not in self.bad]
                if not alternatives:break
                a=alternatives[0]
            before=ctx.progress(); revision=obs.get('revision'); oldobs=obs
            try:
                obs=self.game_call(ctx.act,a['id'])
                if obs is None:obs=self.game_call(ctx.observe)
            except (BudgetExceeded,PlanningStopped):
                raise
            except Exception as exc:
                if ctx.done:break
                self.failure='Action raised '+str(exc)[:160];self.bad.add(key(a));self.rejects+=1
                obs=self.game_call(ctx.observe)
                if self.rejects>=6:break
                continue
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
                    State(new,check=self.check_work).road_length()>=needed and after.get('longest_road_owner')!=color):
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


# Common random numbers compare alternative investments on the same dice worlds.
# This private RNG is unrelated to engine seed or live random state.
import random
import math

_SCENARIOS = None

def dice_worlds():
    global _SCENARIOS
    if _SCENARIOS is None:
        rng = random.Random(217031)
        _SCENARIOS = [[(rng.randint(1,6)+rng.randint(1,6),rng.random())
                       for _ in range(40)] for _ in range(28)]
    return _SCENARIOS


def integer_fund(hand, cost, rates, bank=None, production=None):
    """Pay a complete project with whole maritime lots, preserving its inputs.
    Returns the residual hand, or None. Forecast bank availability is uncertain;
    current execution always uses State.trades and the authoritative legal menu.
    """
    h=list(hand)
    need=sum(max(0,cost[i]-h[i]) for i in range(5))
    if sum(max(0,h[i]-cost[i])//rates[i] for i in range(5)) < need:
        return None
    if bank is not None and any(max(0,cost[i]-h[i])>bank[i] for i in range(5)):
        return None
    prod=production or (1,1,1,1,1)
    for j in range(5):
        while h[j]<cost[j]:
            donors=[i for i in range(5) if h[i]-cost[i]>=rates[i]]
            if not donors:return None
            # Preserve hard-to-replace cards when more than one exact funding
            # sequence exists. This is a forecast tie-break, not a live trade.
            i=min(donors,key=lambda i:rates[i]/(0.35+prod[i])+0.03*cost[i])
            h[i]-=rates[i];h[j]+=1
    return tuple(h[i]-cost[i] for i in range(5))


def discard_to_goal(hand, cost, rates):
    h=list(hand)
    for _ in range(sum(h)//2):
        candidates=[i for i in range(5) if h[i]>0]
        # Excess cards first; then preserve the least covered required inputs.
        i=max(candidates,key=lambda i:(h[i]-cost[i],h[i]/rates[i],h[i]))
        h[i]-=1
    return h


class DiceState(State):
    """Two-investment, partially observed economic scenario scheduler."""
    def __init__(self,o,check=None):
        super().__init__(o,check=check)
        self._portfolio_cache={}
        self.roll_yield=[[0]*5 for _ in range(13)]
        self.blocked_yield=[[0]*5 for _ in range(13)]
        for n in self.mine:
            mult=2 if self.v[n].get('building')=='CITY' else 1
            self.add_site_yields(n,mult,self.roll_yield,self.blocked_yield)

    def add_site_yields(self,n,mult,yields,blocked):
        for hi in self.v[n].get('adjacent_hexes',[]):
            h=self.h[hi];r=h.get('terrain');number=h.get('number')
            if r in R and number in PIPS:
                target=blocked if h.get('robber') else yields
                target[number][R.index(r)]+=mult

    def _reward(self,p):
        # Preserve the parent's strategic valuation; the experiment changes
        # timing, resource discreteness, and the alternatives being compared.
        return p['value']

    def _candidates(self,projects):
        # Include the best of every kind, all currently funded projects, and
        # enough routes for the first investment to have a meaningful choice.
        selected=[];seen=set()
        def add(p):
            if p['id'] not in seen:selected.append(p);seen.add(p['id'])
        for p in projects:
            if p['ready']:add(p)
        for typ in ('build_city','build_settlement','buy_development_card','award'):
            pool=[p for p in projects if p['type']==typ]
            for p in pool[:3 if typ=='build_settlement' else 2]:add(p)
        for p in projects[:8]:add(p)
        return selected[:14]

    def _followups(self,first,pool):
        choices=[]
        pieces=self.me.get('pieces',{})
        remaining_cities=pieces.get('cities',4)-(first['type']=='build_city')
        remaining_settlements=pieces.get('settlements',5)-(first['type']=='build_settlement')+(first['type']=='build_city')
        remaining_roads=pieces.get('roads',15)-len(first['path'])
        for q in pool:
            if q['type']=='build_city' and remaining_cities<=0:continue
            if q['type']=='build_settlement' and remaining_settlements<=0:continue
            if q['type']=='award' and first['type']=='award':continue
            if q['type']=='buy_development_card' and self.o.get('development_cards_remaining',0)-(first['type']=='buy_development_card')<=0:continue
            if q['id']==first['id'] and q['type']!='buy_development_card':continue
            # Neighboring settlement projects conflict under the distance rule.
            if (first['type']=='build_settlement' and q['type']=='build_settlement'
                and q['target'] in self.v[first['target']].get('neighbors',[])):continue
            qq=dict(q);qq['cost']=list(q['cost'])
            # Pay shared access roads once, not once for each settlement.
            if first['path'] and q['path']:
                paid={tuple(sorted(e)) for e in first['path']}
                common=sum(tuple(sorted(e)) in paid for e in q['path'])
                qq['cost'][0]-=common;qq['cost'][1]-=common
            else:common=0
            if len(q['path'])-common>remaining_roads:continue
            choices.append(qq)
        if first['type']=='build_settlement' and self.me.get('pieces',{}).get('cities',4)>0:
            n=first['target'];val=6+0.60*self.pip_value(n)+max(0,self.vp-4)*2
            choices.append(dict(id='future_city:'+str(n),type='build_city',target=n,
                                path=[],cost=list(COST['build_city']),value=val))
        return choices

    def _forecast(self,p,pool):
        completed=[];payoffs=[];discarded=0;spent_now=0
        cost=p['cost'];initial_rate=self.rate
        followups=self._followups(p,pool)
        # Candidate selection at the second investment uses only resources
        # already realized in that scenario; it never sees future dice.
        for world in dice_worlds():
            self.check()
            hand=list(self.hand);rate=list(initial_rate);current_bank=list(self.bank)
            yields=[list(x) for x in self.roll_yield]
            blocked=[list(x) for x in self.blocked_yield]
            released=False;first_time=None;payoff=0.0;second=None
            for rd in range(11):
                current=p if first_time is None else second
                if current is not None:
                    left=integer_fund(hand,current['cost'],rate,
                                      current_bank if rd==0 else None,self.production)
                    if left is not None:
                        payoff+=self._reward(current)*(0.84**rd)
                        if rd==0:current_bank=[current_bank[i]+hand[i]-left[i] for i in range(5)]
                        hand=list(left)
                        if first_time is None:
                            first_time=rd;completed.append(rd)
                            if rd==0:spent_now+=1
                            if p['type'] in ('build_settlement','build_city'):
                                self.add_site_yields(p['target'],1,yields,blocked)
                            if p['type']=='build_settlement':
                                port=self.v[p['target']].get('port')
                                if port=='ANY':rate=[min(x,3) for x in rate]
                                elif port in R:rate[R.index(port)]=2
                            def next_score(q):
                                if integer_fund(hand,q['cost'],rate,None,self.production) is not None:
                                    return self._reward(q)
                                prod=[sum(PIPS.get(n,0)*(yields[n][i]+0.5*blocked[n][i])/9 for n in PIPS) for i in range(5)]
                                # Only a selection heuristic. Actual completion
                                # and scoring below remain integer/dice based.
                                t=0
                                for t in range(1,11):
                                    projected=[int(hand[i]+prod[i]*t) for i in range(5)]
                                    if integer_fund(projected,q['cost'],rate,None,prod) is not None:break
                                return self._reward(q)*0.84**t
                            second=max(followups,key=next_score) if followups else None
                            # A second already-funded investment can occur in
                            # the same real turn, without another dice round.
                            if second is not None:
                                left=integer_fund(hand,second['cost'],rate,
                                                  current_bank if rd==0 else None,self.production)
                                if left is not None:
                                    payoff+=self._reward(second)*(0.84**rd)
                                    break
                        else:break
                if rd==10:break
                current=p if first_time is None else second
                goal=current['cost'] if current else [0]*5
                for roll,u in world[rd*4:rd*4+4]:
                    if roll==7:
                        if sum(hand)>7:
                            old=sum(hand);hand=discard_to_goal(hand,goal,rate)
                            discarded+=old-sum(hand)
                        # Approximation: current block may be removed after a
                        # seven. New enemy targeting and theft are not known.
                        if u<0.5:released=True
                    else:
                        for i in range(5):
                            hand[i]=min(19,hand[i]+yields[roll][i]+(blocked[roll][i] if released else 0))
            # Future output outside our bounded horizon is deliberately not
            # called a completed investment.
            payoffs.append(payoff)
        n=len(payoffs); completed.sort()
        return dict(utility=round(sum(payoffs)/n,3),
                    completion_probability=round(len(completed)/n,3),
                    p_complete_2=round(sum(t<=2 for t in completed)/n,3),
                    median_rounds=completed[len(completed)//2] if completed else 11,
                    p80_rounds=completed[min(len(completed)-1,int(n*0.8))] if len(completed)>=math.ceil(n*0.8) else 11,
                    discard_cards=round(discarded/n,2))

    def projects(self,mode='balanced',preferred=None):
        ck=(mode,preferred,getattr(self,'road_awards_enabled',True))
        if ck in self._portfolio_cache:return self._portfolio_cache[ck]
        raw=super().projects(mode,preferred)
        if self.setup or not raw:return raw
        pool=self._candidates(raw)
        for p in pool:
            self.check()
            stats=self._forecast(p,pool)
            p['forecast']=stats
            p['fluid_eta']=p['eta'];p['eta']=stats['median_rounds']
            score=stats['utility']
            # Astra's advice is a bounded strategic bias, never an invented
            # action or an override of exact affordability.
            if mode=='cities' and p['type']=='build_city':score*=1.10
            if mode=='expand' and p['type']=='build_settlement':score*=1.10
            if mode=='army' and p['type']=='buy_development_card':score*=1.10
            if p['id']==preferred:score*=1.08
            p['score']=round(score,3)
        # Unsimulated alternatives remain visible but do not outrank sampled
        # investments on incomparable continuous-ETA score units.
        for p in raw:
            if 'forecast' not in p:p['score']=0
        result=sorted(raw,key=lambda p:p['score'],reverse=True)
        self._portfolio_cache[ck]=result
        return result


class ScenarioHarness(Harness):
    def __init__(self,ctx):
        super().__init__(ctx)
        self.intent=None;self.intent_started=None;self.intent_limit=None
        self.last_forecast_failure=-99;self.waits=0
        self.retired=[];self.discard_phase=None;self.discard_goal=None

    def ask(self,s,projects,reason,opening=False):
        if not opening:
            reason+='; discrete forecasts assume future bank liquidity, no theft, uncertain robber release, and fixed public routes. '
            reason+='Current investment intent='+str(self.intent)+'; deadline_turn='+str(self.intent_limit)+'. '
            reason+='Choose an investment, considering completion probability, public rival threats, and forecast failure evidence. '
            reason+='Forecast utility compares TWO investments on common dice worlds; median_rounds is not a guarantee. '
        return super().ask(s,projects,reason,opening)

    def choose(self,s,projects):
        if 'discard_resource' in s.bytype:
            phase=(s.o.get('turn_number'),s.o.get('turn_owner'))
            if self.discard_phase!=phase:
                self.discard_phase=phase
                self.discard_goal=list(projects[0]['cost'] if projects else COST['build_city'])
            goal=self.discard_goal
            def surplus(a):
                i=R.index(a['parameters'])
                return (s.hand[i]-goal[i],s.hand[i]/s.rate[i],s.hand[i])
            return max(s.bytype['discard_resource'],key=surplus)
        self.discard_phase=None
        # Let the parent's authoritative compiler handle all interruptions,
        # tactical cards, immediate wins, and already-started transactions.
        action=super().choose(s,projects)
        if s.setup or not projects or 'end_turn' not in s.bytype or self.failure:return action
        if action['type'] in ('play_knight_card','play_monopoly','play_year_of_plenty',
                              'play_road_building','move_robber','discard_resource','roll'):
            return action
        if s.vp>=9 and action['type'] in ('build_city','build_settlement'):return action
        top=projects[0]
        if top['ready']:
            self.intent=None;self.waits=0
            return action
        # super.choose may have just marked an affordable alternate as active.
        # Only a transaction begun on an earlier accepted action is protected.
        committed=getattr(self,'_executing_transaction',None)
        if committed and committed==self.active_project:return action
        turn=s.o.get('turn_number',0)
        # Do not wait on a forecast with weak probability of completion or an
        # already-failed waiting deadline. Fresh advice can still choose builds.
        f=top.get('forecast',{})
        if f.get('p_complete_2',0)<0.5:return action
        if turn-self.last_forecast_failure<8:return action
        funded=[p for p in projects if p['ready']]
        best_funded=max((p['score'] for p in funded),default=0)
        if top['score']<=1.05*best_funded:return action
        # An opponent on nine public VP gets an immediate conversion response;
        # forecasts of future turns are irrelevant if there may be no next turn.
        if max([0]+[p.get('public_vp',0) for p in s.o.get('opponents',[])])>=9:return action
        if self.intent!=top['id']:
            if self.intent is None:
                self.intent_started=turn
                self.intent_limit=turn+4*max(2,min(4,f.get('p80_rounds',4)))
            self.intent=top['id']
        self.active_project=None
        self.waits+=1
        self.ctx.log('scenario_wait',project=top['id'],forecast=f,
                     best_funded_utility=best_funded,deadline_turn=self.intent_limit,
                     progress=self.ctx.progress())
        return s.bytype['end_turn'][0]

    def poll(self,s,projects):
        # This hook runs after the fresh public state and portfolio exist.
        self._executing_transaction=self.active_project if self.active_turn==s.o.get('turn_number') else None
        if self.intent and not s.setup and 'end_turn' in s.bytype:
            turn=s.o.get('turn_number',0)
            exists=next((p for p in projects if p['id']==self.intent),None)
            if exists is None or (turn>=self.intent_limit and not exists['ready']):
                self.failure=('Discrete investment forecast failed: '+self.intent+
                              (' target disappeared' if exists is None else ' not fundable by predicted deadline')+
                              '; progress '+json.dumps(self.ctx.progress()))
                self.ctx.log('forecast_failure',reason=self.failure)
                self.last_forecast_failure=turn;self.intent=None;self.preferred=None
        self.retired=[f for f in self.retired if not f.done()]
        if self.failure and self.pending is not None and not self.pending.done() and not self.retired:
            # An obsolete queued request cannot receive new failure evidence.
            # Admit at most one replacement call; retired model work is ignored
            # and the context cancels it automatically when the episode closes.
            self.retired.append(self.pending);self.pending=None
            self.ctx.log('supersede_stale_planner',reason=self.failure,
                         progress=self.ctx.progress())
        return super().poll(s,projects)

    def verify(self,old,new,a,progress):
        previous_failure=self.failure
        super().verify(old,new,a,progress)
        if a['type'] in ('build_city','build_settlement','buy_development_card'):
            self.intent=None;self.waits=0;self.active_project=None
        if self.failure and self.failure!=previous_failure:self.intent=None


def build(ctx):
    return ScenarioHarness(ctx)
