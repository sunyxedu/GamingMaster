"""Whole-turn public-state macro search for Catan.

This controller is deliberately deterministic.  It treats the adaptor's legal
list as authoritative, but keeps a small resource and route model so that
maritime trades are made for a build that is actually reachable.  The model is
kept out of the low-level loop: a stale or malformed model answer cannot block
a turn-game episode.
"""
from __future__ import annotations

import json
import math
import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, replace

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


def _trade_parts(a):
    p = a.get("parameters")
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return None, None
    # Catanatron represents 4:1 as four inputs plus output, and a 3:1/2:1
    # trade with None placeholders.  The output is always the final value.
    out = p[-1] if isinstance(p[-1], str) else None
    ins = [x for x in p[:-1] if isinstance(x, str) and x in RES]
    return (ins[0] if ins else None), out


def _trade_score(o, a, acts):
    """Value a trade by the best build it enables after the trade."""
    selfv = o.get("self", {}) if isinstance(o, dict) else {}
    have = selfv.get("resources", {}) if isinstance(selfv.get("resources", {}), dict) else {}
    ins, out = _trade_parts(a)
    if not ins or out not in RES or have.get(ins, 0) < len([x for x in (a.get("parameters") or [])[:-1] if x == ins]):
        return -1000.0
    nxt = {r: int(have.get(r, 0) or 0) for r in RES}
    for _ in range(len([x for x in (a.get("parameters") or [])[:-1] if x == ins])):
        nxt[ins] -= 1
    nxt[out] += 1
    best = -1.0
    for typ, cost in COSTS.items():
        deficit = sum(max(0, n - nxt.get(r, 0)) for r, n in cost.items())
        # A trade completing a VP action is strong; a road is useful only when
        # it opens a productive settlement site.
        weight = {"build_city": 8.0, "build_settlement": 7.0, "build_road": 2.5, "buy_development_card": 1.2}[typ]
        best = max(best, weight - 1.8 * deficit)
    # Avoid cycling a surplus into a resource that does not reduce any deficit.
    return best


def _robber(o, acts):
    b, hs, vertices = _board(o)
    opp = {q.get("id") for q in o.get("opponents", []) if isinstance(q, dict)}
    scored = []
    for a in acts:
        if a.get("type") != "move_robber":
            continue
        p = a.get("parameters")
        coord = p[0] if isinstance(p, list) and p else None
        victim = p[1] if isinstance(p, list) and len(p) > 1 else None
        h = next((x for x in hs if isinstance(x, dict) and x.get("coordinate") == coord), {})
        pressure = PIPS.get(h.get("number"), 0) * (2.0 if victim in opp else 0.15)
        for vid in h.get("vertices", []):
            v = vertices.get(vid, {})
            if v.get("owner") in opp:
                pressure += 1.5 if v.get("building") == "CITY" else 0.75
        for q in o.get("opponents", []):
            if isinstance(q, dict) and q.get("id") == victim:
                pressure += min(6, float(q.get("resource_count", 0) or 0)) * 0.25
                pressure += float(q.get("public_vp", 0) or 0) * 0.15
        scored.append((pressure, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)



# Commands contain semantic parameters, never observation-local IDs.
def cmd(kind, param=None):
    if isinstance(param, list):
        param = tuple(param)
    return kind, param


def action_command(a):
    p = a.get('parameters')
    if a['type'] == 'maritime_trade':
        ins = [r for r in p[:-1] if r in RES]
        return cmd('maritime_trade', (RES.index(ins[0]), len(ins), RES.index(p[-1])))
    if a['type'] == 'build_road':
        p = tuple(sorted(p))
    return cmd(a['type'], p)


COST = {'build_city': (0, 0, 0, 2, 3),
        'build_settlement': (1, 1, 1, 1, 0),
        'build_road': (1, 1, 0, 0, 0),
        'buy_development_card': (0, 0, 1, 1, 1)}


def vec(d):
    return tuple(int(d.get(r, 0) or 0) for r in RES)


@dataclass(frozen=True)
class State:
    hand: tuple
    bank: tuple
    roads: frozenset
    settlements: frozenset
    cities: frozenset
    rates: tuple
    vp: int
    award: bool
    deck_value: float = 0.0


class Search:
    def __init__(self, obs, ctx=None):
        self.obs = obs
        self.ctx = ctx
        self.me = obs['self']
        self.color = self.me.get('id', 'RED')
        self.acts = _acts(obs)
        self.vertices = {v['id']: v for v in obs['board']['vertices']}
        self.hexes = {h['id']: h for h in obs['board']['hexes']}
        self.edges = {tuple(sorted(e['id'])): e.get('owner') for e in obs['board']['edges']}
        self.adj = defaultdict(list)
        for e in self.edges:
            u, v = e
            self.adj[u].append((v, e))
            self.adj[v].append((u, e))
        self.enemy = frozenset(v for v, d in self.vertices.items() if d.get('owner') not in (None, self.color))
        self.occupied = frozenset(v for v, d in self.vertices.items() if d.get('owner'))
        settlements = frozenset(v for v, d in self.vertices.items() if d.get('owner') == self.color and d.get('building') == 'SETTLEMENT')
        cities = frozenset(v for v, d in self.vertices.items() if d.get('owner') == self.color and d.get('building') == 'CITY')
        roads = frozenset(e for e, owner in self.edges.items() if owner == self.color)
        ports = self.me.get('ports', [])
        rates = [2 if r in ports else 3 if None in ports else 4 for r in RES]
        for a in self.acts:
            if a['type'] == 'maritime_trade':
                i, rate, _ = action_command(a)[1]
                rates[i] = min(rates[i], rate)
        self.awards = obs.get('awards', {})
        self.root = State(vec(self.me.get('resources', {})), vec(obs.get('bank_counts', {r:19 for r in RES})), roads,
                          settlements, cities, tuple(rates), int(self.me.get('true_vp', self.me.get('public_vp', 0))),
                          self.awards.get('longest_road_owner') == self.color)
        self.road_cache = {}
        self.geometry_cache = {}
        self.production_cache = {}
        self.payment_cache = {}
        self.longest_cache = {}
        self.stop = time.monotonic() + 1.4
        self.threshold = max(5, int(self.awards.get('longest_road_length', 0)) + (1 if self.awards.get('longest_road_owner') else 0))
        self.horizon = max(0.4, (10 - self.root.vp) / 3.0)
        self.root_map = {action_command(a): a for a in self.acts if a['type'] not in ('move_robber',)}
        self.dev_allowed = any(a['type'] == 'buy_development_card' for a in self.acts) or not self.afford(self.root.hand, COST['buy_development_card'])
        self.vertex_prod = {}
        for v, d in self.vertices.items():
            p = [0.0] * 5
            for hid in d.get('adjacent_hexes', []):
                h = self.hexes.get(hid, {})
                if h.get('terrain') in RES:
                    p[RES.index(h['terrain'])] += PIPS.get(h.get('number'), 0) / 9 * (0.45 if h.get('robber') else 1)
            self.vertex_prod[v] = tuple(p)

    def alive(self):
        return time.monotonic() < self.stop and not (self.ctx is not None and self.ctx.done)

    @staticmethod
    def afford(hand, cost):
        return all(h >= c for h, c in zip(hand, cost))

    def production(self, s):
        key = s.settlements, s.cities
        if key not in self.production_cache:
            self.production_cache[key] = tuple(sum(self.vertex_prod[v][i] for v in s.settlements) +
                                               2 * sum(self.vertex_prod[v][i] for v in s.cities) for i in range(5))
        return self.production_cache[key]

    def longest(self, roads):
        if roads in self.longest_cache:
            return self.longest_cache[roads]
        adj = defaultdict(list)
        for i, (u, v) in enumerate(sorted(roads)):
            if u in self.enemy or v in self.enemy:
                continue
            adj[u].append((v, 1 << i)); adj[v].append((u, 1 << i))
        def walk(v, used):
            return max((1 + walk(w, used | bit) for w, bit in adj[v] if not used & bit), default=0)
        result = max((walk(v, 0) for v in list(adj)), default=0)
        self.longest_cache[roads] = result
        return result

    def geometry(self, s):
        key = s.roads, s.settlements | s.cities
        if key in self.geometry_cache:
            return self.geometry_cache[key]
        owned = s.settlements | s.cities
        occupied = self.enemy | owned
        sites = {v for v in self.vertices if v not in occupied and all(w not in occupied for w, e in self.adj[v])}
        starts = owned | {v for e in s.roads for v in e if v not in self.enemy}
        best = {v: (0, ()) for v in starts}
        heap = [(0, v, ()) for v in starts]
        heapq.heapify(heap)
        while heap:
            n, v, path = heapq.heappop(heap)
            if best.get(v) != (n, path) or v in self.enemy:
                continue
            for w, e in self.adj[v]:
                if w in self.enemy or self.edges[e] not in (None, self.color):
                    continue
                new = path if e in s.roads else path + (e,)
                if len(new) > 3:
                    continue
                if w not in best or (len(new), new) < best[w]:
                    best[w] = len(new), new
                    heapq.heappush(heap, (len(new), w, new))
        routes = {v: path for v, (n, path) in best.items() if v in sites and len(path) <= 15-len(s.roads)}
        self.geometry_cache[key] = routes
        return routes

    def extensions(self, s, roads):
        starts = s.settlements | s.cities | {v for e in roads for v in e if v not in self.enemy}
        return [e for e, owner in self.edges.items() if owner is None and e not in roads
                and not any(v in self.enemy for v in e) and any(v in starts for v in e)]

    def award_paths(self, s):
        if s.award or len(s.roads) >= 15:
            return []
        if s.roads in self.road_cache:
            return self.road_cache[s.roads]
        length = self.longest(s.roads)
        # Most early road networks cannot justify a four-road award investment.
        limit = min(4 if s.vp >= 7 else 3, 15-len(s.roads))
        if length + limit < self.threshold and len(s.roads) + limit < self.threshold:
            return []
        beam = [(s.roads, ())]
        seen = {s.roads}
        found = []
        for _ in range(limit):
            next_beam = []
            for roads, path in beam:
                for e in self.extensions(s, roads):
                    new = roads | {e}
                    if new in seen:
                        continue
                    seen.add(new)
                    n = self.longest(new)
                    np = path + (e,)
                    if n >= self.threshold:
                        found.append(np)
                    else:
                        next_beam.append((n, sum(sum(self.vertex_prod[v]) for v in e), new, np))
            if found or not self.alive():
                break
            next_beam.sort(key=lambda z:(z[0], z[1]), reverse=True)
            beam = [(z[2], z[3]) for z in next_beam[:18]]
        found.sort(key=lambda p:(len(p), p))
        self.road_cache[s.roads] = found[:4]
        return found[:4]

    def goals(self, s, include_award=True):
        out = []
        if len(s.cities) < 4:
            out.extend((('build_city', v), COST['build_city'], (cmd('build_city', v),)) for v in sorted(s.settlements))
        if len(s.settlements) < 5:
            for v, path in self.geometry(s).items():
                cost = (1+len(path), 1+len(path), 1, 1, 0)
                sequence = tuple(cmd('build_road', e) for e in path) + (cmd('build_settlement', v),)
                out.append((('build_settlement', v), cost, sequence))
        if include_award:
            for path in self.award_paths(s):
                out.append((('road_award', None), (len(path),len(path),0,0,0), tuple(cmd('build_road', e) for e in path)))
        if self.dev_allowed:
            out.append((('buy_development_card', None), COST['buy_development_card'], (cmd('buy_development_card'),)))
        return out

    def payments(self, s, cost):
        """Exact integer trades toward an entire macro; never sell its reserves."""
        key = s.hand, s.bank, s.rates, cost
        if key in self.payment_cache:
            return self.payment_cache[key]
        queue = deque([(s.hand, s.bank, ())]); seen = {s.hand}; found = []
        # Cost deficits give an acyclic exchange problem: every exchange fills
        # one needed card using only resources beyond the macro's full reserve.
        while queue and len(seen) < 1600:
            hand, bank, path = queue.popleft()
            deficits = [i for i in range(5) if hand[i] < cost[i]]
            if not deficits:
                found.append((hand, bank, path))
                continue
            if sum(max(0,h-c)//rate for h,c,rate in zip(hand,cost,s.rates)) < sum(cost[i]-hand[i] for i in deficits):
                continue
            # The next deficit resource is fixed: independent input choices
            # still enumerate all economically different payment portfolios.
            j = next((i for i in deficits if bank[i] > 0), None)
            if j is None:
                continue
            for i in range(5):
                rate = s.rates[i]
                if i == j or hand[i]-cost[i] < rate:
                    continue
                nh=list(hand); nb=list(bank)
                nh[i]-=rate; nh[j]+=1; nb[i]+=rate; nb[j]-=1
                nh=tuple(nh)
                if nh in seen:
                    continue
                seen.add(nh)
                queue.append((nh,tuple(nb),path+(cmd('maritime_trade',(i,rate,j)),)))
        prod=self.production(s)
        found.sort(key=lambda z:(len(z[2]), -sum(h/(0.5+p) for h,p in zip(z[0],prod))))
        self.payment_cache[key]=found[:3]
        return found[:3]

    def transition(self, s, sequence):
        """Deterministic private-hand/public-board transition, no engine calls."""
        for typ, p in sequence:
            if s.vp >= 10:
                break
            if typ == 'maritime_trade':
                i,rate,j=p; h=list(s.hand); b=list(s.bank)
                if h[i]<rate or b[j]<1:
                    return None
                h[i]-=rate; h[j]+=1; b[i]+=rate; b[j]-=1
                s=replace(s,hand=tuple(h),bank=tuple(b)); continue
            cost=COST.get(typ)
            if cost is None or not self.afford(s.hand,cost):
                return None
            s=replace(s,hand=tuple(h-c for h,c in zip(s.hand,cost)),bank=tuple(b+c for b,c in zip(s.bank,cost)))
            if typ=='build_city':
                if p not in s.settlements or len(s.cities)>=4: return None
                s=replace(s,settlements=s.settlements-{p},cities=s.cities|{p},vp=s.vp+1)
            elif typ=='build_settlement':
                if p not in self.geometry(s) or self.geometry(s)[p] or len(s.settlements)>=5: return None
                rates=list(s.rates); port=self.vertices[p].get('port')
                if port in RES: rates[RES.index(port)]=2
                s=replace(s,settlements=s.settlements|{p},rates=tuple(rates),vp=s.vp+1)
            elif typ=='build_road':
                if p not in self.extensions(s,s.roads) or len(s.roads)>=15: return None
                roads=s.roads|{p}
                award=s.award or self.longest(roads)>=self.threshold
                s=replace(s,roads=roads,award=award,vp=s.vp+(2 if award and not s.award else 0))
            elif typ=='buy_development_card':
                knights=int(self.me.get('development_cards',{}).get('KNIGHT',0))
                needed=max(3,int(self.awards.get('largest_army_size',0))+1)-int(self.me.get('played_knights',0))
                army=0 if self.awards.get('largest_army_owner')==self.color else 1.6/max(2,needed+knights)
                s=replace(s,deck_value=s.deck_value+1.3+army)
        return s

    def value(self,s):
        if s.vp>=10:
            return 1000+100*(s.vp-10)
        prod=self.production(s)
        # Diminishing returns value coverage and ports without enforcing a
        # particular economy. This is a heuristic, not a waiting-time theorem.
        econ=sum(math.log1p(2.5*p) for p in prod)
        econ+=sum(max(0,p-0.5)*(4/r-1)*0.45 for p,r in zip(prod,s.rates))
        caps=(2,2,2,3,3)
        hand=0.0
        for h,p,cap,rate in zip(s.hand,prod,caps,s.rates):
            weight=0.40+0.18/(0.5+p)
            hand+=weight*(min(h,cap)+max(0,h-cap)/rate)
        if sum(s.hand)>7:
            hand-=0.07*(sum(s.hand)-7)
        return 5.0*s.vp+self.horizon*econ+hand+s.deck_value

    def eta(self,s,cost):
        prod=self.production(s)
        def feasible(t):
            stock=[int(h+p*t+1e-8) for h,p in zip(s.hand,prod)]
            missing=sum(max(0,c-h) for h,c in zip(stock,cost))
            spare=sum(max(0,h-c)//r for h,c,r in zip(stock,cost,s.rates))
            return spare>=missing
        if feasible(0):return 0.0
        lo,hi=0.,32.
        if not feasible(hi):return 64.
        for _ in range(10):
            mid=(lo+hi)/2
            if feasible(mid):hi=mid
            else:lo=mid
        return hi

    def reserve(self,goals):
        s=self.root
        ranked=[]
        for target,cost,sequence in goals:
            typ,v=target
            points=2 if typ=='road_award' else 0.3 if typ=='buy_development_card' else 1
            economy=sum(self.vertex_prod[v]) if v is not None else 0
            value=5*points+self.horizon*economy
            if s.vp+points>=10:value+=40
            eta=self.eta(s,cost)
            ranked.append((value/(1+eta+0.18*len(sequence)),target,cost,sequence,eta))
        if not ranked:return None
        ranked.sort(key=lambda z:z[0],reverse=True)
        _,target,cost,sequence,eta=ranked[0]
        best=None
        for a in self.acts:
            if a['type']!='maritime_trade':continue
            command=action_command(a);i,rate,j=command[1]
            if s.hand[j]>=cost[j] or s.hand[i]-rate<min(s.hand[i],cost[i]):continue
            ns=self.transition(s,(command,))
            if ns is None:continue
            ne=self.eta(ns,cost)
            if ne<eta-0.03 or (sum(s.hand)>7 and ne<=eta+0.001):
                key=(-ne,self.value(ns))
                if best is None or key>best[0]:best=(key,a)
        if best:return best[1]
        # Invest in a short productive expansion only if the full build is
        # close; all future action IDs still come from the next observation.
        if target[0]=='build_settlement' and len(sequence)>1 and eta<=1.5 and self.root.vp<8:
            a=self.root_map.get(sequence[0])
            if a:return a
        return None

    def choose(self):
        s=self.root
        goals=self.goals(s)
        baseline=self.value(s)
        best_score=baseline
        best_path=()
        beam=[(s,())]
        seen={s:0}
        for depth in range(3):
            children=[]
            for current,prefix in beam:
                for target,cost,builds in (goals if depth==0 else self.goals(current,include_award=True)):
                    for h,b,payment in self.payments(current,cost):
                        path=prefix+payment+builds
                        if not path or path[0] not in self.root_map:continue
                        nxt=self.transition(current,payment+builds)
                        if nxt is None:continue
                        score=self.value(nxt)-0.025*len(path)
                        if score>best_score+1e-8:
                            best_score,best_path=score,path
                        if nxt.vp<10 and target[0]!='buy_development_card' and (nxt not in seen or seen[nxt]>len(path)):
                            seen[nxt]=len(path);children.append((score,nxt,path))
                if not self.alive():break
            if not children or not self.alive():break
            children.sort(key=lambda z:z[0],reverse=True)
            beam=[(z[1],z[2]) for z in children[:14]]
        if best_path:
            return self.root_map[best_path[0]], {'mode':'complete-turn', 'path':best_path,'value_gain':round(best_score-baseline,3)}
        a=self.reserve(goals)
        if a:return a,{'mode':'reserve'}
        return next((a for a in self.acts if a['type']=='end_turn'),self.acts[0]), {'mode':'wait'}


def choose(obs,ctx):
    acts=_acts(obs)
    if not acts:return None,None
    if len(acts)==1:return acts[0],None
    types={a['type'] for a in acts}
    phase=str(obs.get('phase','')).lower()
    if 'initial_settlement' in phase:return _initial_settlement(obs,acts,ctx.memory),None
    if 'initial_road' in phase:return _initial_road(obs,acts),None
    if 'move_robber' in types:return _robber(obs,acts),None
    search=Search(obs,ctx)
    if types=={'build_road'}:
        # Free roads are evaluated as future settlement/award access.
        paths=search.award_paths(search.root)
        for path in paths:
            a=search.root_map.get(cmd('build_road',path[0]))
            if a:return a,{'mode':'free-award'}
        def road_value(a):
            road=tuple(sorted(a['parameters']))
            ns=replace(search.root,roads=search.root.roads|{road})
            return max(((1+sum(search.vertex_prod[v]))/(1+len(p)) for v,p in search.geometry(ns).items()),default=0)+0.10*search.longest(ns.roads)
        return max(acts,key=road_value),{'mode':'free-expansion'}
    knight=next((a for a in acts if a['type']=='play_knight_card'),None)
    roll=next((a for a in acts if a['type']=='roll'),None)
    if roll:
        return knight or roll,None
    action,plan=search.choose()
    if plan.get('mode')=='complete-turn' and plan.get('value_gain',0)>900:
        return action,plan
    # Non-VP card effects are re-observed rather than pretending to know theft
    # or development draws. Plenty is evaluated by the macro search itself.
    if knight:return knight,{'mode':'knight'}
    plenty=[a for a in acts if a['type']=='play_year_of_plenty']
    if plenty:
        goals=search.goals(search.root)
        def value(a):
            h=list(search.root.hand)
            for r in a.get('parameters') or []:
                if r in RES:h[RES.index(r)]+=1
            ns=replace(search.root,hand=tuple(h))
            return max((5*(2 if t[0]=='road_award' else 0.4 if t[0]=='buy_development_card' else 1)/(1+search.eta(ns,c)) for t,c,p in goals),default=0)
        chosen=max(plenty,key=value)
        return chosen,{'mode':'plenty'}
    roadcard=next((a for a in acts if a['type']=='play_road_building'),None)
    if roadcard:return roadcard,{'mode':'road-building'}
    mono=[a for a in acts if a['type']=='play_monopoly']
    if mono:
        estimates=[0.0]*5
        for opp in obs.get('opponents',[]):
            prod=[0.3]*5
            for v,d in search.vertices.items():
                if d.get('owner')==opp.get('id'):
                    mult=2 if d.get('building')=='CITY' else 1
                    for i,p in enumerate(search.vertex_prod[v]):prod[i]+=mult*p
            for i,p in enumerate(prod):estimates[i]+=opp.get('resource_count',0)*p/sum(prod)
        a=max(mono,key=lambda a:estimates[RES.index(a['parameters'])])
        if max(estimates)>=3:return a,{'mode':'monopoly','estimate':round(max(estimates),2)}
    return action,plan


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
                a,plan=choose(obs,ctx)
            except Exception as exc:
                ctx.log('search_fallback',error=type(exc).__name__,detail=str(exc)[:160])
                a=next((a for a in acts if a['type'] in ('roll','end_turn','discard')),acts[0]);plan=None
            if not a or a['id'] not in {x['id'] for x in acts}:a=acts[0]
            if plan:
                ctx.memory['turn_search']=plan
                if plan.get('mode')=='complete-turn':ctx.log('turn_plan',**plan)
            old_count=obs.get('action_count')
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
            if old_count is not None and nxt.get('action_count')==old_count:
                failures+=1
                if failures>=3:return
            else:failures=0
            obs=nxt


def build(ctx):
    return Harness(ctx)
