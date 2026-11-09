"""Public-state, whole-turn beam search. No engine or model imports."""
from __future__ import annotations

import json

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
COSTS = {
    "build_road": {"WOOD": 1, "BRICK": 1},
    "build_settlement": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "build_city": {"ORE": 3, "WHEAT": 2},
    "buy_development_card": {"ORE": 1, "SHEEP": 1, "WHEAT": 1},
}


def _obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


def _acts(o):
    xs = o.get("legal_actions", []) if isinstance(o, dict) else []
    return [a for a in xs if isinstance(a, dict) and a.get("id") is not None]


def _vertex_id(a):
    p = a.get("parameters")
    if isinstance(p, int):
        return p
    if isinstance(p, (list, tuple)):
        for x in p:
            if isinstance(x, int):
                return x
    return None


def _endpoints(a):
    p = a.get("parameters")
    if isinstance(p, (list, tuple)):
        return [x for x in p if isinstance(x, int)]
    return []


def _board(o):
    b = o.get("board", {}) if isinstance(o, dict) else {}
    hs = b.get("hexes", []) if isinstance(b.get("hexes", []), list) else []
    vs = b.get("vertices", []) if isinstance(b.get("vertices", []), list) else []
    return b, hs, {v.get("id"): v for v in vs if isinstance(v, dict)}


def _vertex_score(v, hs, missing=()):
    if not isinstance(v, dict):
        return -100.0
    total = 0.0
    kinds = set()
    for i in v.get("adjacent_hexes", []):
        if not isinstance(i, int) or i < 0 or i >= len(hs):
            continue
        h = hs[i] if isinstance(hs[i], dict) else {}
        terrain, number = h.get("terrain"), h.get("number")
        if not terrain or not isinstance(number, (int, float)) or h.get("robber"):
            continue
        kinds.add(terrain)
        total += PIPS.get(int(number), 0)
        if terrain in ("WHEAT", "ORE"):
            total += 0.55
        if terrain in missing:
            total += 1.4
    total += 0.75 * len(kinds)
    if v.get("port"):
        total += 0.6
    return total


def _my_terrains(o, hs, vertices):
    out = set()
    for v in vertices.values():
        if v.get("owner") == "RED" and v.get("building") in ("SETTLEMENT", "CITY"):
            for i in v.get("adjacent_hexes", []):
                if isinstance(i, int) and 0 <= i < len(hs):
                    t = hs[i].get("terrain") if isinstance(hs[i], dict) else None
                    if t:
                        out.add(t)
    return out


def _initial_settlement(o, acts, memory):
    b, hs, vertices = _board(o)
    # During the snake placement, the second site should complement the first
    # rather than merely duplicate its highest pip intersection.
    old = _my_terrains(o, hs, vertices)
    missing = tuple(t for t in RES if t not in old)
    scored = []
    for a in acts:
        if "settlement" not in str(a.get("type", "")):
            continue
        v = vertices.get(_vertex_id(a), {})
        s = _vertex_score(v, hs, missing)
        if old:
            kinds = set()
            for i in v.get("adjacent_hexes", []):
                if isinstance(i, int) and 0 <= i < len(hs):
                    t = hs[i].get("terrain") if isinstance(hs[i], dict) else None
                    if t:
                        kinds.add(t)
            # Brick is especially valuable for the first expansion and ore
            # prevents an all-wood/sheep opening from becoming trade-locked.
            if "BRICK" in kinds and "BRICK" not in old:
                s += 2.1
            if "ORE" in kinds and "ORE" not in old:
                s += 1.7
            s += 1.1 * len(kinds - old)
        scored.append((s, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


def _initial_road(o, acts):
    _, hs, vertices = _board(o)
    scored = []
    for a in acts:
        if "road" not in str(a.get("type", "")):
            continue
        ends = _endpoints(a)
        s = 0.0
        for e in ends:
            s += _vertex_score(vertices.get(e, {}), hs)
            v = vertices.get(e, {})
            if v.get("building") is None:
                s += 0.5
        scored.append((s, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


# The search below is independent of the simulator and consumes no randomness.
from dataclasses import dataclass, replace
from collections import deque
import math
import time

COST = {'build_road': (1,1,0,0,0), 'build_settlement': (1,1,1,1,0),
        'build_city': (0,0,0,2,3), 'buy_development_card': (0,0,1,1,1)}

@dataclass(frozen=True)
class State:
    hand: tuple
    bank: tuple
    settlements: frozenset
    cities: frozenset
    roads: frozenset
    knights: int
    army: bool
    road_award: bool
    free: int = 0
    played: bool = False
    bonus: float = 0.0
    stop: bool = False


class Search:
    def __init__(self, obs, memory=None):
        self.obs = obs
        me = obs['self']
        self.me = me.get('id', 'RED')
        self.acts = _acts(obs)
        self.vertices = {v['id']: v for v in obs['board']['vertices']}
        self.hexes = {h['id']: h for h in obs['board']['hexes']}
        self.edges = {tuple(sorted(e['id'])): e.get('owner') for e in obs['board']['edges']}
        self.adj = {v: [] for v in self.vertices}
        for u,v in self.edges:
            self.adj[u].append(v); self.adj[v].append(u)
        self.blocked = {v for v,d in self.vertices.items() if d.get('owner') not in (None,self.me)}
        self.opproads = {e for e,c in self.edges.items() if c not in (None,self.me)}
        self.sites = {v for v,d in self.vertices.items() if not d.get('owner')
                      and not any(self.vertices[n].get('owner') for n in self.adj[v])}
        self.pips = {}
        self.fullpips = {}
        for v,d in self.vertices.items():
            p = [0.0]*5; full = [0.0]*5
            for hid in d['adjacent_hexes']:
                h = self.hexes[hid]; r = h.get('terrain')
                if r in RES:
                    i = RES.index(r); q = PIPS.get(h.get('number'),0)
                    full[i] += q
                    # A current robber is temporary, but obstructed production
                    # still deserves a substantial discount in a short horizon.
                    p[i] += q * (0.35 if h.get('robber') else 1.0)
            self.pips[v] = tuple(p); self.fullpips[v] = tuple(full)
        awards = obs.get('awards',{})
        self.awards = awards
        self.baseports = set(me.get('ports',[]))
        self.port_rates = tuple(2 if r in self.baseports else 3 if None in self.baseports else 4 for r in RES)
        # A current legal trade is direct evidence, overriding inferred rates.
        rates = list(self.port_rates)
        for a in self.acts:
            if a['type']=='maritime_trade':
                inp = [x for x in a['parameters'][:-1] if x in RES]
                if inp: rates[RES.index(inp[0])] = min(rates[RES.index(inp[0])],len(inp))
        self.port_rates = tuple(rates)
        onlyroads = {a['type'] for a in self.acts} == {'build_road'}
        self.root = State(tuple(me['resources'].get(r,0) for r in RES),
            tuple(obs.get('bank_counts',{}).get(r,19) for r in RES),
            frozenset(v for v,d in self.vertices.items() if d.get('owner')==self.me and d.get('building')=='SETTLEMENT'),
            frozenset(v for v,d in self.vertices.items() if d.get('owner')==self.me and d.get('building')=='CITY'),
            frozenset(e for e,c in self.edges.items() if c==self.me),
            me.get('played_knights',0), awards.get('largest_army_owner')==self.me,
            awards.get('longest_road_owner')==self.me,
            max(1, (memory or {}).get('free_roads',1)) if onlyroads else 0)
        self.vp0 = float(me.get('true_vp',me.get('public_vp',0)))
        self.road_cache = {}; self.position_cache = {}; self.value_cache = {}
        self.rival_road = max([4, awards.get('longest_road_length',0) if awards.get('longest_road_owner')!=self.me else 0]
            + [self.longest(frozenset(e for e,c in self.edges.items() if c==q['id']),q['id']) for q in obs.get('opponents',[])])
        self.army_target = max(3, 1+max([q.get('played_knights',0) for q in obs.get('opponents',[])] or [0]))
        self.card_actions = [a for a in self.acts if a['type'].startswith('play_')]
        # The observation has resource bank counts but no development-deck
        # count. If an otherwise affordable purchase is absent in a normal
        # build phase, do not plan a later trade toward a phantom purchase.
        can_pay_dev = all(x>=c for x,c in zip(self.root.hand,COST['buy_development_card']))
        normal = not onlyroads and not any(a['type']=='roll' for a in self.acts)
        self.deck_left = not (normal and can_pay_dev and not any(
            a['type']=='buy_development_card' for a in self.acts))
        self.prod0 = self.production(self.root)
        self.prod_scale = max(0.2,(10-self.vp0)/8)
        # Inventory prices are fixed within this search, so transformations
        # cannot manufacture value by changing their own accounting prices.
        self.prices = tuple((1.6 if i in (3,4) else 1.2) + 1.4/(1+self.prod0[i]/3) for i in range(5))
        self.mono = self.monopoly_estimates()

    def production(self,s):
        return tuple(sum(self.pips[v][i] for v in s.settlements) +
                     2*sum(self.pips[v][i] for v in s.cities) for i in range(5))

    def vp(self,s):
        return self.vp0 + len(s.settlements)-len(self.root.settlements) + 2*(len(s.cities)-len(self.root.cities)) + 2*(s.army-self.root.army) + 2*(s.road_award-self.root.road_award)

    def rates(self,s):
        rates = list(self.port_rates)
        for v in (s.settlements | s.cities) - (self.root.settlements | self.root.cities):
            port = self.vertices[v].get('port')
            # Null at an unowned vertex does not identify a generic port.
            if port in RES: rates[RES.index(port)] = 2
        return rates

    def longest(self,roads,color=None,added=frozenset()):
        color = color or self.me
        key = (roads,color,added)
        if key in self.road_cache: return self.road_cache[key]
        blocked = {v for v,d in self.vertices.items() if d.get('owner') not in (None,color)}
        if color != self.me:
            blocked |= added
        graph = {}
        for i,(u,v) in enumerate(sorted(roads)):
            if u in blocked or v in blocked: continue
            graph.setdefault(u,[]).append((v,1<<i)); graph.setdefault(v,[]).append((u,1<<i))
        def trail(v,used):
            return max((1+trail(n,used|bit) for n,bit in graph.get(v,()) if not used&bit),default=0)
        ans = max((trail(v,0) for v in graph),default=0)
        self.road_cache[key]=ans
        return ans

    def owns_road_after(self,s,roads,settlements):
        if s.road_award:
            return True  # Our own additions cannot sever our own road.
        added = (settlements | s.cities) - (self.root.settlements | self.root.cities)
        rival = self.rival_road
        if added:
            rival = max([4] + [self.longest(
                frozenset(e for e,c in self.edges.items() if c==q['id']),
                q['id'],added) for q in self.obs.get('opponents',[])])
        return self.longest(roads) > rival

    def geometry(self,s):
        key=(s.settlements,s.cities,s.roads)
        if key in self.position_cache: return self.position_cache[key]
        own = s.settlements | s.cities
        roots = set(own)
        for e in s.roads: roots.update(e)
        roots -= self.blocked
        valid = self.sites - own
        valid = {v for v in valid if not any(n in own for n in self.adj[v])}
        roads = [e for e,c in self.edges.items() if e not in s.roads and e not in self.opproads
                 and any(v in roots for v in e)]
        connected = valid & roots
        # Distance is number of *new* roads from the whole existing network.
        distance={v:0 for v in roots}; q=deque(roots)
        while q:
            u=q.popleft()
            if u in self.blocked or distance[u]>=3: continue
            for v in self.adj[u]:
                e=tuple(sorted((u,v)))
                if e in self.opproads or v in self.blocked: continue
                d=distance[u]+(e not in s.roads)
                if d<distance.get(v,99):
                    distance[v]=d
                    if e in s.roads:q.appendleft(v)
                    else:q.append(v)
        opportunities=[]
        for v in valid:
            if v in distance:
                quality=1+sum(self.pips[v])/12
                opportunities.append(quality/(1+distance[v])**1.5)
        opportunities.sort(reverse=True)
        option=sum(x*w for x,w in zip(opportunities,(1.0,0.45)))
        ans=(roads,connected,option)
        self.position_cache[key]=ans
        return ans

    def monopoly_estimates(self):
        result=[0.0]*5
        for q in self.obs.get('opponents',[]):
            p=[0.5]*5
            for v,d in self.vertices.items():
                if d.get('owner')==q['id']:
                    mult=2 if d.get('building')=='CITY' else 1
                    for i in range(5):p[i]+=mult*self.fullpips[v][i]
            total=sum(p)
            for i in range(5):result[i]+=q.get('resource_count',0)*p[i]/total
        return result

    def value(self,s):
        if s in self.value_cache:return self.value_cache[s]
        vp=self.vp(s)
        if vp>=10:return 10000+100*vp
        prod=self.production(s)
        economy=sum(0.8*x+2.5*math.sqrt(x) for x in prod)
        # Preserve two wheat / three ore for cities; later surplus has only
        # its bank conversion value. A beam can still spend reserves whenever
        # the complete resulting position justifies it.
        reserve=(2,2,1,2,3)
        inv=0.0
        rates=self.rates(s)
        for i,n in enumerate(s.hand):
            kept=min(n,reserve[i]); extra=max(0,n-kept)
            inv+=self.prices[i]*(kept+0.42*min(extra,2)+0.20*max(0,extra-2))
            if rates[i]<4:inv+=0.20*n*(4/rates[i]-1)
        exposed=max(0,sum(s.hand)-7)
        inv-=0.22*exposed
        option=self.geometry(s)[2] if len(s.settlements)<5 else 0
        # Small progress value for Knights, not a phantom two-point award.
        army_progress=0 if s.army else 1.8*min(s.knights,self.army_target-1)
        result=22*vp+self.prod_scale*economy+inv+3.0*option+army_progress+s.bonus
        self.value_cache[s]=result
        return result

    def generated(self,s):
        if s.stop or self.vp(s)>=10:return []
        roads,sites,_=self.geometry(s)
        out=[]
        def afford(typ):return all(x>=c for x,c in zip(s.hand,COST[typ]))
        if s.free:
            return [('build_road',e) for e in roads] if len(s.roads)<15 else []
        if len(s.cities)<4 and afford('build_city'):
            out.extend(('build_city',v) for v in sorted(s.settlements))
        if len(s.settlements)<5 and afford('build_settlement'):
            out.extend(('build_settlement',v) for v in sorted(sites))
        if len(s.roads)<15 and afford('build_road'):
            out.extend(('build_road',e) for e in roads)
        if afford('buy_development_card') and self.deck_left:
            out.append(('buy_development_card',None))
        rates=self.rates(s)
        for i in range(5):
            if s.hand[i]>=rates[i]:
                for j in range(5):
                    if i!=j and s.bank[j]>0:
                        out.append(('maritime_trade',tuple([RES[i]]*rates[i]+[None]*(4-rates[i])+[RES[j]])))
        if not s.played:
            out.extend((a['type'],a.get('parameters')) for a in self.card_actions)
        return out

    def apply(self,s,typ,p):
        h=list(s.hand); bank=list(s.bank)
        if typ in COST and not (typ=='build_road' and s.free):
            cost=COST[typ]
            if any(x<c for x,c in zip(h,cost)):return None
            h=[x-c for x,c in zip(h,cost)];bank=[x+c for x,c in zip(bank,cost)]
        changes={'hand':tuple(h),'bank':tuple(bank)}
        if typ=='build_city':
            if p not in s.settlements or len(s.cities)>=4:return None
            changes.update(settlements=s.settlements-{p},cities=s.cities|{p})
        elif typ=='build_settlement':
            if len(s.settlements)>=5:return None
            settlements=s.settlements|{p}
            changes.update(settlements=settlements,
                           road_award=self.owns_road_after(s,s.roads,settlements))
        elif typ=='build_road':
            e=tuple(sorted(p)); roads=s.roads|{e}
            if len(roads)>15:return None
            award=self.owns_road_after(s,roads,s.settlements)
            changes.update(roads=roads,road_award=award,free=max(0,s.free-1))
        elif typ=='maritime_trade':
            inp=[r for r in p[:-1] if r in RES]
            if not inp or p[-1] not in RES:return None
            i=RES.index(inp[0]);j=RES.index(p[-1]);n=len(inp)
            if h[i]<n or bank[j]<1:return None
            h[i]-=n;h[j]+=1;bank[i]+=n;bank[j]-=1
            changes.update(hand=tuple(h),bank=tuple(bank))
        elif typ=='buy_development_card':
            # Prior expected value only: do not invent a card or continue
            # through an unknown draw. The actual next observation replans.
            need=max(1,self.army_target-s.knights)
            army_value=0 if s.army else min(7.0,14.0/need)
            value=4.4+army_value+1.5
            changes.update(bonus=s.bonus+value,stop=True)
        elif typ=='play_knight_card':
            if s.played:return None
            blocked=sum(sum(self.fullpips[v])-sum(self.pips[v]) for v in s.settlements)+2*sum(sum(self.fullpips[v])-sum(self.pips[v]) for v in s.cities)
            changes.update(knights=s.knights+1,army=s.army or s.knights+1>=self.army_target,
                           played=True,bonus=s.bonus+1.2+0.5*blocked,stop=True)
        elif typ=='play_year_of_plenty':
            if s.played:return None
            for r in p:
                if r not in RES:continue
                i=RES.index(r)
                if bank[i]<=0:return None
                bank[i]-=1;h[i]+=1
            changes.update(hand=tuple(h),bank=tuple(bank),played=True,bonus=s.bonus-1.0)
        elif typ=='play_road_building':
            if s.played:return None
            changes.update(free=min(2,15-len(s.roads)),played=True,bonus=s.bonus-0.7)
        elif typ=='play_monopoly':
            if s.played or p not in RES:return None
            i=RES.index(p); amount=self.mono[i]
            # Unknown transfer is a leaf option, never a guaranteed win.
            h[i]+=int(amount)
            changes.update(hand=tuple(h),played=True,bonus=s.bonus-2.0,stop=True)
        elif typ=='end_turn':
            changes.update(stop=True)
        else:return None
        return replace(s,**changes)

    def choose(self,ctx=None):
        start=time.monotonic(); deadline=start+0.8
        best=None;bestvalue=-1e30;bestpath=None
        beam=[]; seen={}; expanded=0
        # Force a current legal root, never an inferred future action.
        for a in self.acts:
            s=self.apply(self.root,a['type'],a.get('parameters'))
            if s is None:continue
            val=self.value(s)-0.10
            path=((a['type'],a.get('parameters')),)
            if val>bestvalue:best,bestvalue,bestpath=a,val,path
            if not s.stop and self.vp(s)<10:
                beam.append((val,s,a,path));seen[s]=1
        beam.sort(key=lambda x:x[0],reverse=True);beam=beam[:28]
        for depth in range(2,9):
            if not beam or time.monotonic()>deadline or (ctx is not None and ctx.done):break
            nxt=[]
            for _,s,first,path in beam:
                for typ,p in self.generated(s):
                    ns=self.apply(s,typ,p)
                    if ns is None or seen.get(ns,99)<=depth:continue
                    seen[ns]=depth;expanded+=1
                    val=self.value(ns)-0.10*depth
                    newpath=path+((typ,p),)
                    if val>bestvalue:best,bestvalue,bestpath=first,val,newpath
                    if not ns.stop and self.vp(ns)<10:nxt.append((val,ns,first,newpath))
                if expanded>=5500 or time.monotonic()>deadline:break
            if expanded>=5500 or time.monotonic()>deadline:break
            nxt.sort(key=lambda x:x[0],reverse=True);beam=nxt[:28]
        if best is None:best=next((a for a in self.acts if a['type']=='end_turn'),self.acts[0])
        return best,{'path':bestpath,'nodes':expanded,'value':round(bestvalue,3),'search_s':round(time.monotonic()-start,4)}


def robber(obs,acts):
    me=obs['self'].get('id','RED')
    verts={v['id']:v for v in obs['board']['vertices']}
    opp={q['id']:q for q in obs.get('opponents',[])}
    def value(a):
        p=a.get('parameters') or [];coord=p[0] if p else None;victim=p[1] if len(p)>1 else None
        h=next((h for h in obs['board']['hexes'] if h['coordinate']==coord),{})
        score=0.0
        for v in h.get('vertices',[]):
            d=verts[v];owner=d.get('owner');mult=2 if d.get('building')=='CITY' else 1
            pip=PIPS.get(h.get('number'),0)*mult
            if owner==me:score-=4*pip
            elif owner in opp:
                q=opp[owner];threat=q.get('public_vp',0)+0.25*q.get('development_count',0)
                score+=pip*(0.7+0.12*threat)
        if victim in opp:score+=1+0.2*min(5,opp[victim].get('resource_count',0))
        return score
    return max(acts,key=value)


def choose(obs,memory,ctx=None):
    acts=_acts(obs)
    if not acts:return None,None
    if len(acts)==1:return acts[0],None
    types={a['type'] for a in acts};phase=str(obs.get('phase','')).lower()
    if 'initial_settlement' in phase:return _initial_settlement(obs,acts,memory),None
    if 'initial_road' in phase:return _initial_road(obs,acts),None
    if 'move_robber' in types:return robber(obs,acts),None
    if 'discard' in types:return next(a for a in acts if a['type']=='discard'),None
    if 'roll' in types:
        knight=next((a for a in acts if a['type']=='play_knight_card'),None)
        if knight:
            s=Search(obs,memory);ns=s.apply(s.root,'play_knight_card',None)
            blocked=any(h.get('robber') and any(s.vertices[v].get('owner')==s.me for v in h.get('vertices',[])) for h in obs['board']['hexes'])
            if s.vp(ns)>=10 or blocked:return knight,None
        return next(a for a in acts if a['type']=='roll'),None
    search=Search(obs,memory)
    return search.choose(ctx)


class Harness:
    def __init__(self,ctx):self.ctx=ctx
    def run_episode(self,game):
        ctx=self.ctx
        if ctx.done:return
        try:obs=_obj(ctx.observe())
        except Exception:return
        failures=0
        while not ctx.done and not obs.get('terminal',False):
            acts=_acts(obs)
            if not acts:return
            try:
                a,info=choose(obs,ctx.memory,ctx)
                if info:
                    ctx.memory['sequence_search']=info
                    ctx.log('sequence_search',**info)
            except Exception as exc:
                ctx.log('search_fallback',error=type(exc).__name__,message=str(exc)[:160])
                a=next((a for a in acts if a['type'] in ('roll','end_turn','discard')),acts[0])
            if ctx.done:return
            if not a or a['id'] not in {x['id'] for x in acts}:a=acts[0]
            before=obs.get('action_count')
            try:
                nxt=_obj(ctx.act(a['id']))
                if not nxt and not ctx.done:nxt=_obj(ctx.observe())
            except Exception:
                if ctx.done:return
                failures+=1
                if failures>=3:return
                try:obs=_obj(ctx.observe())
                except Exception:return
                continue
            if before is not None and nxt.get('action_count')==before:
                failures+=1
                if failures>=3:return
            else:
                failures=0
                if a['type']=='play_road_building':ctx.memory['free_roads']=2
                elif a['type']=='build_road':ctx.memory['free_roads']=max(0,ctx.memory.get('free_roads',0)-1)
                elif a['type']=='end_turn':ctx.memory['free_roads']=0
            obs=nxt


def build(ctx):return Harness(ctx)
