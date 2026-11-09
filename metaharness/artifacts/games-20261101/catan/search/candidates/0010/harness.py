"""Discrete stochastic opening economics over the leader's tactical compiler.
Only public observations; forecasts are approximate and actions use live legal IDs.
"""
from __future__ import annotations
import json
import heapq
import re
import math
import random

class OpeningDraft:
    """Public-board snake draft scenarios; never a copy of the hidden game.

    Opponent preferences are deliberately varied. Forecasts are planning features,
    not assertions that the bots will choose those intersections.
    """
    def __init__(self, s, stop=None):
        self.s = s
        self.stop = stop or (lambda: False)
        self.forecast_occupied = set()
        self.cache = {}
        rng = random.Random(17431)
        self.dice_paths = [[rng.randint(1,6)+rng.randint(1,6) for _ in range(48)] for _ in range(20)]
        self.block = {n: {n, *v.get('neighbors', [])} for n, v in s.v.items()}
        self.occupied = {n for n, v in s.v.items() if v.get('owner')}
        self.blocked = set().union(*(self.block[n] for n in self.occupied)) if self.occupied else set()
        self.order = s.o.get('seat_order', [])
        self.menus = s.bytype.get('build_settlement', [])

    @staticmethod
    def throughput(p, cost, rates):
        # Continuous maritime relaxation: resource rate and trade losses both
        # matter. It is a sustainable build rate, not an expected waiting time.
        lo, hi = 0.0, sum(p) / max(1, sum(cost))
        for _ in range(16):
            mid = (lo + hi) / 2
            deficit = sum(max(0, mid*c - x) for x, c in zip(p, cost))
            credit = sum(max(0, x - mid*c)/r for x, c, r in zip(p, cost, rates))
            if credit >= deficit: lo = mid
            else: hi = mid
        return lo

    def pair(self, first, second):
        if self.stop(): return {'value':-100, 'first_growth_round_p75':13, 'third_site_round_p75':13}
        expansions = self.expansion(first, second)
        target = expansions[0] if expansions else None
        sig = (first, second, target[1] if target else None, target[2] if target else None)
        if sig in self.cache:
            return dict(self.cache[sig], expansion_options=len(expansions),
                        expansion_path=target[3] if target else [])
        result = self.rollout_pair(first, second, expansions)
        self.cache[sig] = result
        return result

    def profile(self, nodes):
        profile = [[0]*5 for _ in range(13)]
        for n in nodes:
            for hi in self.s.v[n].get('adjacent_hexes', []):
                h = self.s.h[hi]
                if h.get('terrain') in R and h.get('number') in PIPS:
                    profile[h['number']][R.index(h['terrain'])] += 1
        return profile

    def expansion(self, first, second):
        # Public graph only. One setup road can point along this route for free.
        # During the actual second placement the first road is already observed.
        blocked = self.blocked | self.block[first] | self.block[second]
        for n in self.forecast_occupied: blocked |= self.block[n]
        routes = []
        for root in (first, second):
            q = [(0, root, [])]; seen = {root}
            while q:
                d, n, path = q.pop(0)
                if d >= 3: continue
                for z in self.s.v[n].get('neighbors', []):
                    if z in seen or z in self.forecast_occupied or self.s.v[z].get('owner') not in (None, self.s.color): continue
                    edge = tuple(sorted((n,z)))
                    owner = self.s.edges.get(edge)
                    if owner not in (None, self.s.color): continue
                    seen.add(z); newpath = path+[edge]
                    q.append((d+1, z, newpath))
                    if z in blocked: continue
                    paid = sum(self.s.edges.get(e)!=self.s.color for e in newpath)
                    if root == second or not self.s.mine: paid = max(0, paid-1)
                    pips = [self.s.site[first][i]+self.s.site[second][i] for i in range(5)]
                    gain = sum(x*(1+2/(1+pips[i])) for i,x in enumerate(self.s.site[z]))
                    port = self.s.v[z].get('port')
                    if port == 'ANY': gain += 4
                    elif port in R: gain += min(6,pips[R.index(port)]*.6)
                    routes.append((gain/(1+.5*paid),z,paid,[list(e) for e in newpath]))
        return sorted(routes,reverse=True)

    @staticmethod
    def pay(hand, cost, rates):
        # Integer trades: four fractional quarter-surpluses do not buy a card.
        deficit = sum(max(0,c-h) for h,c in zip(hand,cost))
        credit = sum(max(0,h-c)//r for h,c,r in zip(hand,cost,rates))
        if credit < deficit: return None
        h = list(hand)
        for j in range(5):
            while h[j] < cost[j]:
                donors = [i for i in range(5) if h[i]-cost[i]>=rates[i]]
                if not donors: return None
                i = min(donors,key=lambda i:(rates[i],-(h[i]-cost[i])))
                h[i] -= rates[i]; h[j] += 1
        return [h[i]-cost[i] for i in range(5)]

    def rollout_pair(self, first, second, expansions):
        s = self.s
        pips = [s.site[first][i]+s.site[second][i] for i in range(5)]
        ports = {s.v[n].get('port') for n in (first,second)}
        rates = [2 if r in ports else 3 if 'ANY' in ports else 4 for r in R]
        start = [0]*5
        for hi in s.v[second].get('adjacent_hexes',[]):
            r = s.h[hi].get('terrain')
            if r in R: start[R.index(r)] += 1
        # Both a production expansion and a city-first plan are tested. This
        # is a forecast of investment speed, NOT a binding game action queue.
        target = expansions[0] if expansions else None
        nodes = [first,second]+([target[1]] if target else [])
        profiles = [self.profile([n]) for n in nodes]
        base = self.profile([first,second])
        settlement_cost = (1+target[2],1+target[2],1,1,0) if target else None
        orders = [(0,1,2,3),(1,0,2,3),(2,0,1,3),(2,1,0,3),
                  (0,2,1,3),(1,2,0,3)] if target else [(0,1),(1,0)]
        best = None
        for order in orders:
            scores=[]; first_times=[]; third_times=[]
            for dice in self.dice_paths:
                hand=list(start); prod=[row[:] for row in base]; rate=list(rates)
                stage=0; utility=0.; first_time=13; third_time=13
                for t,number in enumerate(dice,1):
                    cost = settlement_cost if stage<len(order) and order[stage]==2 else COST['build_city']
                    if number==7:
                        if sum(hand)>7:
                            for _ in range(sum(hand)//2):
                                i=max((i for i in range(5) if hand[i]>0),
                                      key=lambda i:(hand[i]-cost[i],hand[i],-i))
                                hand[i]-=1
                    else:
                        row=prod[number]
                        hand=[hand[i]+row[i] for i in range(5)]
                    if t%4: continue
                    while stage<len(order):
                        task=order[stage]
                        cost=settlement_cost if task==2 else COST['build_city']
                        funded=self.pay(hand,cost,rate)
                        if funded is None: break
                        hand=funded; stage+=1
                        roundno=t/4
                        utility+=10*(.94**roundno)
                        if first_time==13: first_time=roundno
                        if task==2:
                            third_time=roundno
                            port=s.v[target[1]].get('port')
                            rate=[min(rate[i],2 if r==port else 3 if port=='ANY' else 4) for i,r in enumerate(R)]
                        profile=profiles[2 if task in (2,3) else task]
                        for n in PIPS:
                            for i in range(5):prod[n][i]+=profile[n][i]
                    if stage==len(order): break
                # Small residual value avoids treating stalled-but-rich as zero.
                utility+=.12*sum(hand[i]/rate[i] for i in range(5))
                scores.append(utility);first_times.append(first_time);third_times.append(third_time)
            ordered=sorted(scores); n=len(scores)
            value=.7*sum(scores)/n+.3*sum(ordered[:max(1,n//4)])/max(1,n//4)
            detail={'value':value,'first_growth_round_p75':sorted(first_times)[int(.75*(n-1))],
                    'third_site_round_p75':sorted(third_times)[int(.75*(n-1))],
                    'growth_by_round12':sum(t<13 for t in first_times)/n,
                    'third_site_by_round12':sum(t<13 for t in third_times)/n,
                    'forecast_order':list(order)}
            if best is None or value>best['value']:best=detail
        # Production volume remains a small tie-break, not a fractional bank.
        best['value']=round(best['value']+.15*sum(pips),3)
        best.update(pips=pips,starting_resources=start,expansion_options=len(expansions),
                    expansion_target=target[1] if target else None,
                    expansion_path=target[3] if target else [],
                    paid_roads=target[2] if target else None)
        return best

    def opponents_between(self):
        s = self.s
        if len(s.mine) or s.color not in self.order: return []
        i = self.order.index(s.color)
        return self.order[i+1:] + list(reversed(self.order[i+1:]))

    def forecast(self, first, scenario):
        s = self.s
        blocked = self.blocked | self.block[first]
        holdings = {c: [n for n,v in s.v.items() if v.get('owner') == c] for c in self.order}
        predicted = []
        for j, color in enumerate(self.opponents_between()):
            old = [sum(s.site[n][i] for n in holdings[color]) for i in range(5)]
            options = [n for n in s.v if n not in blocked]
            if not options: break
            def score(n):
                p = s.site[n]
                weights = ([1,1,1,1,1], [1,.9,.85,1.35,1.45],
                           [1.25,1.2,.85,1.2,1], [.9,.8,1,1.5,1.2])[scenario % 4]
                value = sum(x*w for x,w in zip(p, weights))
                value += sum(min(x, max(0, 4-old[i]))*.35 for i,x in enumerate(p))
                port = s.v[n].get('port')
                value += 1.0 if port == 'ANY' else .25*(old[R.index(port)]+p[R.index(port)]) if port in R else 0
                # Two variants per preference permit close alternatives rather
                # than pretending one guessed response is exact.
                value += .35*math.sin((n+1)*(scenario+3)+(j+1)*7)
                return value
            ranked = sorted(options, key=score, reverse=True)
            n = ranked[0]
            if scenario >= 4 and len(ranked)>1 and score(ranked[0])-score(ranked[1]) < .8:
                n = ranked[1]
            holdings[color].append(n); blocked |= self.block[n]
            predicted.append(n)
        self.forecast_occupied = set(predicted)
        options = [n for n in s.v if n not in blocked]
        if not options: return -100, None, predicted
        second = max(options, key=lambda n: self.pair(first,n)['value'])
        return self.pair(first,second)['value'], second, predicted

    def rank(self):
        s = self.s; results = []
        for a in self.menus:
            if self.stop(): break
            self.forecast_occupied = set()
            n = a['parameters']
            if s.mine:
                detail = self.pair(s.mine[0], n)
                value = detail['value']; second = n; scenarios = []
            else:
                forecasts = [self.forecast(n,k) for k in range(8)]
                vals = sorted(x[0] for x in forecasts)
                value = .65*sum(vals)/len(vals) + .35*sum(vals[:2])/2
                best = sorted(forecasts, key=lambda x:x[0])[len(forecasts)//2]
                second = best[1]
                self.forecast_occupied = set(best[2])
                detail = self.pair(n,second) if second is not None else {}
                scenarios = sorted(set(x[1] for x in forecasts if x[1] is not None))
            results.append(dict(**a, draft_score=round(value,3), pips=s.site[n],
                                port=s.v[n].get('port'), likely_second=second,
                                second_alternatives=scenarios, economy=detail))
        return sorted(results, key=lambda x:x['draft_score'], reverse=True)


R = ('WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE')
COST = {'build_road': (1,1,0,0,0), 'build_settlement': (1,1,1,1,0),
        'build_city': (0,0,0,2,3), 'buy_development_card': (0,0,1,1,1)}
PIPS = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}

def key(a):
    return (a['type'], json.dumps(a.get('parameters'), sort_keys=True))

def parse(text):
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

class ResourceLedger:
    """Exact aggregate totals; uncertain per-player estimates constrained by them.

    No unseen hand is read. Events change a soft estimate, and matrix scaling
    reconciles it with public player hand sizes and the resource bank.
    """
    def __init__(self):
        self.hands={}; self.prev=None; self.last_event=-1
        self.aggregate=None; self.valid=False; self.error=None; self.history_gap=False

    def update(self,s):
        o=s.o; ids=[p['id'] for p in o.get('opponents',[])]
        total=[19-s.bank[i]-s.hand[i] for i in range(5)]
        counts={p['id']:p.get('resource_count',0) for p in o.get('opponents',[])}
        self.valid=(all(r in o.get('bank_counts',{}) for r in R) and
                    all(0<=x<=19 for x in total) and sum(total)==sum(counts.values()))
        if not self.valid:
            self.error='Bank conservation mismatch; discard exact aggregate and use soft estimates'
            self.aggregate=None
        else:self.aggregate=total; self.error=None
        hands={c:list(self.hands.get(c,[0.25+x for x in s.raw.get(c,[0]*5)])) for c in ids}
        old=self.prev
        fresh=[e for e in o.get('events',[]) if e.get('index',-1)>self.last_event]
        self.history_gap=bool(o.get('events_truncated') or
                              (old and fresh and fresh[0].get('index',-1)>self.last_event+1))
        if self.history_gap:
            # A missed build changes production topology. Reset soft beliefs
            # instead of replaying incomplete events against an obsolete map.
            hands={c:[.25+x for x in s.raw.get(c,[0]*5)] for c in ids}
            old=None
        vertices={n:dict(v) for n,v in (old.v.items() if old else s.v.items())}
        hexes=old.h if old else s.h
        blocked=next((n for n,h in hexes.items() if h.get('robber')),None)
        for e in o.get('events',[]):
            idx=e.get('index',-1)
            if idx<=self.last_event:continue
            self.last_event=idx
            typ=e.get('type'); c=e.get('player'); p=e.get('parameters')
            # Only use chronological production when the previous graph exists.
            if typ=='roll' and old and isinstance(p,list) and len(p)==2:
                number=sum(p)
                for h in hexes.values():
                    r=h.get('terrain')
                    if h.get('number')!=number or h['id']==blocked or r not in R:continue
                    j=R.index(r)
                    for n in h.get('vertices',[]):
                        v=vertices[n]; owner=v.get('owner')
                        if owner in hands:hands[owner][j]+=2 if v.get('building')=='CITY' else 1
            if typ in ('build_settlement','build_city') and p in vertices:
                vertices[p]['owner']=c
                vertices[p]['building']='CITY' if typ=='build_city' else 'SETTLEMENT'
            if c in hands:
                if typ in COST and not s.setup and old and not old.setup:
                    for i,v in enumerate(COST[typ]):hands[c][i]=max(0,hands[c][i]-v)
                elif typ=='maritime_trade' and isinstance(p,list) and p:
                    for r in p[:-1]:
                        if r in R:hands[c][R.index(r)]=max(0,hands[c][R.index(r)]-1)
                    if p[-1] in R:hands[c][R.index(p[-1])]+=1
                elif typ=='play_year_of_plenty' and isinstance(p,list):
                    for r in p:
                        if r in R:hands[c][R.index(r)]+=1
                elif typ=='discard_resource':
                    if p in R:hands[c][R.index(p)]=max(0,hands[c][R.index(p)]-1)
                    else:
                        z=sum(hands[c]);hands[c]=[v*max(0,z-1)/z if z else 0 for v in hands[c]]
            if typ=='play_monopoly' and p in R:
                j=R.index(p); taken=sum(v[j] for k,v in hands.items() if k!=c)
                for k in hands:
                    if k!=c:hands[k][j]=0
                if c in hands:hands[c][j]+=taken
            if typ=='move_robber' and isinstance(p,list) and len(p)==2:
                blocked=next((h['id'] for h in hexes.values() if h.get('coordinate')==p[0]),blocked)
                victim=p[1]; stolen=e.get('result')
                if stolen in R:
                    j=R.index(stolen)
                    if c in hands:hands[c][j]+=1
                    if victim in hands:hands[victim][j]=max(0,hands[victim][j]-1)
                elif victim in hands:
                    z=sum(hands[victim]); moved=[v/z if z else 0.2 for v in hands[victim]]
                    hands[victim]=[max(0,v-m) for v,m in zip(hands[victim],moved)]
                    if c in hands:hands[c]=[v+m for v,m in zip(hands[c],moved)]
        # Positive smoothing deliberately prevents unobserved trades/discards
        # or event clipping from being mistaken for certain zero holdings.
        for c in ids:
            prior=s.raw.get(c,[0]*5)
            hands[c]=[max(0,v)+0.03+0.03*prior[i] for i,v in enumerate(hands[c])]
        for _ in range(50 if self.valid else 1):
            for c in ids:
                z=sum(hands[c]);hands[c]=[v*counts[c]/z if z else counts[c]/5 for v in hands[c]]
            if self.valid:
                for j in range(5):
                    z=sum(hands[c][j] for c in ids)
                    for c in ids:hands[c][j]=hands[c][j]*total[j]/z if z else total[j]/max(1,len(ids))
        self.hands=hands; self.prev=s
        return self

    def report(self):
        return {'aggregate_exact':self.aggregate,'resource_order':R,
                'individual_estimates_NOT_known':{c:[round(v,2) for v in h] for c,h in self.hands.items()},
                'conservation_valid':self.valid,'history_gap_reset':self.history_gap,'error':self.error}


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx; self.mode='balanced'; self.preferred=None; self.memo=''
        self.calls=0; self.last_call_turn=-999; self.last_vp=-1; self.last_threat=0
        self.failure=None; self.rejects=0; self.last_score_turn=0; self.last_score=-1
        self.bad=set(); self.last_revision=None
        self.pending=None; self.pending_revision=None; self.retired_planners=[]
        self.active_project=None; self.active_turn=None; self.road_awards_enabled=True
        self.ledger=ResourceLedger(); self.accounting_enabled=True; self.target_enemy=None
        self.last_ledger_valid=True
        self.draft_revision=None; self.draft_choices=[]; self.draft_expected=[]
        self.opening_forecast=None; self.opening_checked=set()
        self.opening_road=[]

    def opening_choices(self,s):
        if self.draft_revision != s.o.get('revision'):
            self.draft_choices=OpeningDraft(s,lambda:self.ctx.done).rank()
            self.draft_revision=s.o.get('revision')
            if len(s.mine)==1 and self.draft_expected:
                live={a['parameters'] for a in s.bytype.get('build_settlement',[])}
                if not live.intersection(self.draft_expected):
                    self.failure=('All forecast complementary sites lost: '+str(self.draft_expected)+
                                  '; current progress '+json.dumps(self.ctx.progress()))
            self.ctx.log('opening_rollouts',revision=s.o.get('revision'),
                         choices=self.draft_choices[:8],forecast_failure=self.failure)
        return self.draft_choices

    def check_opening(self,s):
        if s.setup or not self.opening_forecast: return
        turn=s.o.get('turn_number',0); forecast=self.opening_forecast
        # Army/hidden VP cannot conceal a missed production-growth milestone.
        buildings=sum(2 if s.v[n].get('building')=='CITY' else 1 for n in s.mine)
        milestones=[('first_growth',buildings>2),('third_site',len(s.mine)>2)]
        for name,complete in milestones:
            if name in self.opening_checked: continue
            if complete:
                self.opening_checked.add(name); continue
            predicted=forecast.get(name+'_round_p75',13)
            # 13 is censored: fewer than 75% completed within twelve rounds.
            due=4*predicted+4
            if turn>=due:
                self.opening_checked.add(name); self.preferred=None
                label='missed predicted timing' if predicted<=12 else 'still incomplete beyond rollout horizon'
                self.failure=('Opening milestone '+name+' '+label+' at turn '+str(turn)+
                              '; forecast '+json.dumps(forecast)+'; progress '+json.dumps(self.ctx.progress()))
                self.ctx.log('opening_forecast_failed',reason=self.failure)

    def release_stale_planner(self,s):
        # An old setup request can remain queued while an entire local game
        # runs. On actual failure evidence, send the CURRENT situation to
        # Astra rather than letting that old question suppress all recovery.
        self.retired_planners=[f for f in self.retired_planners if not f.done()]
        if not self.failure or self.pending is None or self.pending.done(): return
        if self.pending_revision==s.o.get('revision') or self.retired_planners: return
        budget=self.ctx.budget.remaining()
        if self.calls>=12 or budget.get('system2_calls',0)<=2 or budget.get('wall_s',0)<380:return
        self.retired_planners.append(self.pending)
        self.pending=None
        self.ctx.log('stale_planner_replaced_for_recovery',old_revision=self.pending_revision,
                     current_revision=s.o.get('revision'),failure=self.failure,
                     progress=self.ctx.progress())
        # At most one retired model request plus one active request. Both are
        # ctx-managed inference tasks and are cancelled when the episode ends.

    def ask(self,s,projects,reason,opening=False):
        ctx=self.ctx; budget=ctx.budget.remaining()
        if self.pending is not None:
            return self.poll(s,projects)
        if self.calls>=12 or budget.get('system2_calls',0)<=1 or budget.get('wall_s',0)<380: return None
        cities=[dict(vertex=n,pips=s.site[n],port=s.v[n].get('port')) for n in s.mine]
        options=[]
        if opening:
            options=self.opening_choices(s)
        payload={'reason':reason,'phase':s.o.get('phase'),'turn':s.o.get('turn_number'),
                 'self':s.me,'opponents':s.o.get('opponents'),'awards':s.o.get('awards'),
                 'production_per_round':dict(zip(R,[round(x,2) for x in s.production])),
                 'opponent_production':{c:[round(x,2) for x in p] for c,p in s.raw.items() if c!=s.color},
                 'resource_order':R,'our_sites':cities,'mode':self.mode,'prior_plan':self.memo,
                 'progress':ctx.progress(),'failure':self.failure,'resource_ledger':self.ledger.report(),
                 'target_enemy':self.target_enemy,'opening_forecast':self.opening_forecast,
                 'projects':projects[:12],'opening_choices':options,
                 'recent_events':s.o.get('events',[])[-12:]}
        if not opening:
            payload['legal_actions']=s.legal
            payload['expansion_sites']=[{'vertex':n,'pips':s.site[n],'path':path,'port':s.v[n].get('port')} for n,path in sorted(s.routes.items(),key=lambda x:s.pip_value(x[0]),reverse=True)[:10]]
        prompt=("You manage a Catan economic planner against three strong AlphaBeta bots. Reach 10 true VP first. "
          "Choose a coherent route to victory from code-computed projects. Roads without useful endpoints are wasteful; "
          "ore/wheat city production compounds, development cards give army and hidden VP. Army incumbent wins ties. "
          "Production/ETA are approximate expected values, not guaranteed rolls. Individual opponent hands are uncertain; bank-derived aggregate resource totals are exact only when conservation_valid. Use the ledger to select an economic bottleneck and target_enemy color. "
          "Code executes forced actions, robber/discards, exact maritime trades preserving project cost, and builds, refreshing legality after each. "
          "It replans locally each step. Your project preference is a modest persistent bonus, not an unconditional build order. "
          "Use mode cities, expand, army, or balanced; project must be an offered project id (or null). "
          "For setup choose a legal action ID using opening_choices. draft_score evaluates pairs after varied opponent drafts. Pair economy is a 20-path, twelve-round dice rollout with integer maritime exchanges, seven-discards, and sequential city/third-site investments. It includes starting resources and compounding production. It excludes future opponent blocking, stealing, bank shortages and moving robber, so treat its timings as optimistic forecasts. forecast_order uses 0/1=upgrade first/second site, 2=third settlement plus paid roads, 3=upgrade third site. High pips are not enough: prioritize feasible early production growth and a surviving complement. Prefer the ranked choice unless a concrete board/forecast problem justifies an override. "
          "When failure evidence is present diagnose it and revise mode/project; optionally select one current legal action ID to recover. "
          "Reply ONLY compact JSON: {\"action\":null,\"mode\":\"balanced\",\"project\":null,\"target_enemy\":null,\"plan\":\"under 35 words\"}. "
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
        ans=parse(result.text)
        if isinstance(ans.get('target_enemy'),str) and ans['target_enemy'] in {p['id'] for p in s.o.get('opponents',[])}:
            self.target_enemy=ans['target_enemy']
        if ans.get('mode') in ('cities','expand','army','balanced'): self.mode=ans['mode']
        ids={p['id'] for p in projects}
        if isinstance(ans.get('project'),str) and ans['project'] in ids: self.preferred=ans['project']
        if isinstance(ans.get('plan'),str): self.memo=ans['plan'][:600]
        self.ctx.memory['strategy']={'mode':self.mode,'project':self.preferred,'plan':self.memo}
        if not allow_action:return None
        matches=re.findall(r'r\d+:a\d+',str(ans.get('action','')))
        if not matches: matches=re.findall(r'r\d+:a\d+',result.text)
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
        if self.opening_road:
            a=s.find('build_road',self.opening_road[0])
            if a:
                self.opening_road=[]
                return a
        baseline=s.routes
        def score(a):
            routes=s.find_routes([a['parameters']])
            improved=[(n,p) for n,p in routes.items() if len(p)<len(baseline.get(n,[0]*99))]
            return max([0]+[s.pip_value(n)/(1+len(p)) for n,p in improved])
        return max(s.bytype.get('build_road',s.legal),key=score)

    def robber(self,s):
        players={p['id']:p for p in s.o.get('opponents',[])}
        lead=max([0]+[p.get('public_vp',0) for p in players.values()])
        # Shadow values are a soft opportunity-cost estimate, not a claim to
        # know a hidden hand. The aggregate remains exact and is never split
        # into purportedly exact individual hands.
        shadow={}
        for c,p in players.items():
            hand=self.ledger.hands.get(c,[0]*5); prod=s.raw.get(c,[0]*5)
            weights=[]
            for i in range(5):
                scarcity=1/(.45+prod[i])
                urgency=max([0]+[1.0/(1+sum(max(0,cost[j]-hand[j]) for j in range(5)))
                                  for cost in COST.values() if cost[i]>hand[i]])
                weights.append(min(3.5,.8+scarcity*.35+urgency))
            shadow[c]=weights
        myshadow=[1/(.65+s.production[i]) for i in range(5)]
        def score(a):
            coord,victim=a.get('parameters',[None,None])
            h=next((h for h in s.h.values() if h.get('coordinate')==coord),{})
            r=h.get('terrain'); i=R.index(r) if r in R else 0
            pips=PIPS.get(h.get('number'),0); value=0
            for n in h.get('vertices',[]):
                v=s.v[n]; c=v.get('owner'); mult=2 if v.get('building')=='CITY' else 1
                if c==s.color:value-=pips*mult*(3+myshadow[i])
                elif c in players:
                    p=players[c]; vp=p.get('public_vp',0)
                    threat=1+vp*.18+min(1,p.get('development_count',0)*.15)
                    if vp>=7:threat*=1.7
                    if c==self.target_enemy and vp>=lead-2:threat*=1.25
                    # Removing the only wheat/ore source matters more than
                    # removing one of several interchangeable port inputs.
                    value+=pips*mult*shadow[c][i]*threat
            if victim in players and players[victim].get('resource_count',0)>0:
                hand=self.ledger.hands.get(victim,[1]*5); z=sum(hand)
                usefulness=sum(v*myshadow[j] for j,v in enumerate(hand))/max(.01,z)
                deny=sum(v*shadow[victim][j] for j,v in enumerate(hand))/max(.01,z)
                value+=3*usefulness+deny*(1+players[victim].get('public_vp',0)*.1)
            return value
        a=max(s.bytype['move_robber'],key=score)
        self.ctx.log('ledger_robber',action=a,utility=round(score(a),2),target_enemy=self.target_enemy)
        return a

    def tactical_card(self,s,projects):
        """Choose a one-card-per-turn use by the projects it funds now."""
        b=s.bytype; candidates=[]
        army=s.o.get('awards',{}); army_gain=(army.get('largest_army_owner')!=s.color and
            s.me.get('played_knights',0)+1>=max(3,army.get('largest_army_size',0)+1))
        if 'play_knight_card' in b and army_gain:
            candidates.append((10000 if s.vp>=8 else 32,b['play_knight_card'][0],None,None))
        for typ in ('play_monopoly','play_year_of_plenty'):
            for a in b.get(typ,[]):
                hh=list(s.hand)
                if typ=='play_monopoly':
                    if not self.ledger.valid or not self.accounting_enabled:continue
                    j=R.index(a['parameters']); amount=self.ledger.aggregate[j]
                    hh[j]+=amount
                else:
                    for r in a['parameters']:
                        if r in R:hh[R.index(r)]+=1
                    amount=len(a['parameters'])
                unlock=[]
                for p in projects:
                    if not p['ready'] and s.trades(p['cost'],hh) is not None:
                        win=((p['type'] in ('build_settlement','build_city') and s.vp>=9) or
                             (p['type']=='award' and s.vp>=8))
                        gain=10000 if win else 8+p['value']*.6
                        if p['type']=='buy_development_card':gain*=.5
                        unlock.append((gain,p))
                best=max(unlock,key=lambda x:x[0],default=(0,None))
                # Two cards from Plenty are scarce: demand a newly funded
                # project. Monopoly can justify itself by exact harvest alone.
                if typ=='play_year_of_plenty' and not unlock:continue
                if typ=='play_monopoly' and amount<3 and not unlock:continue
                value=best[0]+amount*1.5
                if typ=='play_monopoly':
                    j=R.index(a['parameters'])
                    for op in s.o.get('opponents',[]):
                        threat=max(0,op.get('public_vp',0)-5)
                        value+=self.ledger.hands.get(op['id'],[0]*5)[j]*threat*.25
                # Roll first unless an exact large harvest or win can vanish.
                if 'roll' in b and best[0]<10000 and amount<6:continue
                candidates.append((value,a,best[1],amount))
        if not candidates:return None
        score,a,p,amount=max(candidates,key=lambda x:x[0])
        if p:self.commit(s,p)
        self.ctx.log('tactical_card',action=a,utility=round(score,2),
                     exact_harvest=amount if a['type']=='play_monopoly' else None,
                     funded_project=p['id'] if p else None)
        return a

    def choose(self,s,projects):
        if not s.setup:
            if s.vp>=9:
                for typ in ('build_settlement','build_city'):
                    if typ in s.bytype:return max(s.bytype[typ],key=lambda a:s.pip_value(a['parameters']))
            special=self.tactical_card(s,projects)
            if special:return special
        return self.base_choose(s,projects)


    def base_choose(self,s,projects):
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
            if 'build_settlement' in b:
                ranked=self.opening_choices(s)
                if ranked:return s.find('build_settlement',ranked[0]['parameters'])
                return max(b['build_settlement'],key=lambda a:s.opening_value(a['parameters']))
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

    def run_episode(self,game):
        ctx=self.ctx; obs=ctx.observe()
        while not ctx.done:
            if not isinstance(obs,dict):obs=ctx.observe()
            if obs.get('terminal'):break
            s=State(obs)
            self.ledger.update(s)
            if not self.ledger.valid and self.last_ledger_valid:
                self.failure=self.ledger.error
            self.last_ledger_valid=self.ledger.valid
            s.road_awards_enabled=self.road_awards_enabled
            if not s.legal:break
            turn=obs.get('turn_number',0)
            if s.setup and 'build_settlement' in s.bytype:self.opening_choices(s)
            self.check_opening(s)
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
            if reason and self.failure:self.release_stale_planner(s)
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
            if s.setup and a.get('type')=='build_settlement':
                chosen=next((x for x in self.draft_choices if x['id']==a['id']),{})
                if not s.mine:self.draft_expected=chosen.get('second_alternatives',[])
                else:self.opening_forecast=chosen.get('economy')
                path=chosen.get('economy',{}).get('expansion_path',[])
                self.opening_road=path if path and a['parameters'] in path[0] else []
            try:
                obs=ctx.act(a['id'])
                if obs is None:obs=ctx.observe()
            except Exception as exc:
                if ctx.done:break
                self.failure='Action raised '+str(exc)[:160];self.bad.add(key(a));self.rejects+=1
                obs=ctx.observe()
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
                    State(new).road_length()>=needed and after.get('longest_road_owner')!=color):
                    self.road_awards_enabled=False
                    failure='Road-award prediction contradicted engine; disabled award-only projects'

        elif typ=='play_monopoly' and p in R and self.accounting_enabled:
            oldhand=old.get('self',{}).get('resources',{})
            expected=19-old.get('bank_counts',{}).get(p,19)-oldhand.get(p,0)
            gained=new.get('self',{}).get('resources',{}).get(p,0)-oldhand.get(p,0)
            if gained!=expected:
                self.accounting_enabled=False
                failure='Monopoly aggregate prediction contradicted: expected '+str(expected)+' got '+str(gained)+'; exact card policy disabled'
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
