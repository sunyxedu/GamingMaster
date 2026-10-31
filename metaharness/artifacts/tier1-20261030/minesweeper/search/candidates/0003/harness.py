"""Exact, low-latency Minesweeper harness with global mine coupling."""
from __future__ import annotations
import math
from collections import defaultdict

def neighbors(r,c,h,w):
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr or dc:
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w: yield rr,cc

def cell(v):
    if isinstance(v,str):
        if v in ("unknown","flagged"): return "h",None
        if v.startswith("revealed:"):
            try:return "r",int(v.split(":",1)[1])
            except Exception:return "r",None
        if v=="mine": return "m",None
    return "h",None

def make_equations(obs,known):
    grid=obs.get("cells") or []; h=int(obs.get("height",len(grid)) or len(grid)); w=int(obs.get("width",len(grid[0]) if grid else 0) or 0)
    hidden,revealed=set(),{}
    for r,row in enumerate(grid[:h]):
        for c,v in enumerate(row[:w]):
            k,n=cell(v)
            if k=="h": hidden.add((r,c))
            elif k=="r" and n is not None: revealed[(r,c)]=n
    known.intersection_update(hidden); base=[]
    for (r,c),n in revealed.items():
        ns=set(neighbors(r,c,h,w)); hs=ns&hidden; rhs=int(n)-len(ns&known); hs-=known
        if hs and 0<=rhs<=len(hs): base.append((frozenset(hs),rhs))
    return hidden,revealed,base,(h,w)

def closure(base,known):
    safe=set(); mines=set(known); work=list(dict.fromkeys(base))
    for _ in range(18):
        changed=False; norm=[]
        for s,rhs in work:
            left=set(s)-mines-safe; need=int(rhs)-len(set(s)&mines)
            if not left or need<0 or need>len(left): continue
            item=(frozenset(left),need)
            if item not in norm:norm.append(item)
            if need==0:
                old=len(safe); safe.update(left); changed|=old!=len(safe)
            elif need==len(left):
                old=len(mines); mines.update(left); changed|=old!=len(mines)
        derived=[]
        for a,na in norm:
            for b,nb in norm:
                if a==b or not a<b: continue
                d,nd=frozenset(b-a),nb-na
                if d and 0<=nd<=len(d):
                    item=(d,nd)
                    if item not in work and item not in derived: derived.append(item)
        if derived: work.extend(derived); changed=True
        if not changed:return safe-mines,mines,norm
    return safe-mines,mines,norm

def components(eqs):
    by=defaultdict(list)
    for s,n in eqs:
        for p in s:by[p].append((s,n))
    seen=set(); out=[]
    for start in by:
        if start in seen:continue
        stack=[start]; cs=set(); got=set()
        while stack:
            p=stack.pop()
            if p in seen:continue
            seen.add(p); cs.add(p)
            for s,n in by[p]: got.add((s,n)); stack.extend(q for q in s if q not in seen)
        out.append((cs,list(got)))
    return out

def enumerate_component(cells,eqs,cap=20):
    ordered=sorted(cells)
    if len(ordered)>cap:return None
    ix={p:i for i,p in enumerate(ordered)}; local=[([ix[p] for p in s],int(n)) for s,n in eqs]
    incidence=[[] for _ in ordered]
    for ei,(inds,_) in enumerate(local):
        for i in inds:incidence[i].append(ei)
    order=sorted(range(len(ordered)),key=lambda i:-len(incidence[i])); assign=[-1]*len(ordered); sums=[0]*len(local); used=[0]*len(local)
    dist=defaultdict(int); counts=defaultdict(lambda:[0]*len(ordered))
    def rec(pos):
        if pos==len(order):
            k=sum(assign); dist[k]+=1; row=counts[k]
            for i,v in enumerate(assign):row[i]+=v
            return
        i=order[pos]
        for v in (0,1):
            ok=True; touched=[]
            for ei in incidence[i]:
                sums[ei]+=v; used[ei]+=1; touched.append(ei); inds,need=local[ei]; rem=len(inds)-used[ei]
                if sums[ei]>need or sums[ei]+rem<need:ok=False
            if ok:assign[i]=v; rec(pos+1)
            for ei in touched:sums[ei]-=v; used[ei]-=1
        assign[i]=-1
    rec(0); return ordered,dict(dist),dict(counts)

def global_rates(hidden,eqs,total_mines,known):
    info=[]; covered=set()
    for cs,ce in components(eqs):
        x=enumerate_component(cs,ce)
        if x is not None and x[1]:info.append(x); covered.update(cs)
    uncon=len(hidden-covered); rem=max(0,int(total_mines)-len(known))
    if not info:
        q=rem/max(1,len(hidden)); return {p:q for p in hidden}
    def combine(a,b):
        out=defaultdict(int)
        for ka,va in a.items():
            for kb,vb in b.items():out[ka+kb]+=va*vb
        return dict(out)
    ds=[x[1] for x in info]; pref=[{0:1}]
    for d in ds:pref.append(combine(pref[-1],d))
    suff=[{} for _ in range(len(ds)+1)]; suff[-1]={0:1}
    for i in range(len(ds)-1,-1,-1):suff[i]=combine(ds[i],suff[i+1])
    def others(i):return combine(pref[i],suff[i+1])
    all_dist=pref[-1]; totalw=0
    for k,count in all_dist.items():
        q=rem-k
        if 0<=q<=uncon:totalw+=count*math.comb(uncon,q)
    rates={}
    if not totalw:
        for ordered,dist,counts in info:
            den=sum(dist.values())
            for j,p in enumerate(ordered):rates[p]=sum(row[j] for row in counts.values())/max(1,den)
        q=rem/max(1,len(hidden)); rates.update({p:q for p in hidden if p not in rates}); return rates
    for i,(ordered,dist,counts) in enumerate(info):
        od=others(i)
        for j,p in enumerate(ordered):
            num=0
            for k,count in dist.items():
                mh=counts[k][j]
                if not mh:continue
                for ko,co in od.items():
                    q=rem-k-ko
                    if 0<=q<=uncon:num+=mh*co*math.comb(uncon,q)
            rates[p]=num/totalw
    if uncon:
        num=0
        for k,count in all_dist.items():
            q=rem-k
            if 0<=q<=uncon:num+=count*math.comb(uncon,q)*q
        q=num/(totalw*uncon)
        for p in hidden-covered:rates[p]=q
    return rates

class Harness:
    def __init__(self,ctx):self.ctx=ctx; self.known=set()
    def run_episode(self,game):
        ctx=self.ctx
        self.known.clear()
        try:obs=game.observe()
        except Exception:return
        first=True
        while not ctx.done:
            if not isinstance(obs,dict):
                try:obs=game.observe()
                except Exception:return
            if obs.get("status")!="playing" or obs.get("won"):return
            grid=obs.get("cells") or []
            if not grid or not grid[0]:return
            hidden,revealed,base,(h,w)=make_equations(obs,self.known)
            if not hidden:return
            if first and int(obs.get("safe_revealed",0) or 0)==0:
                action={"type":"reveal","row":h//2,"col":w//2}; first=False
            else:
                first=False; safe,mines,eqs=closure(base,self.known); self.known.update(mines); safe&=hidden-self.known
                if safe:
                    def sk(p):
                        deg=sum(q in hidden for q in neighbors(p[0],p[1],h,w)); near=sum(q in revealed for q in neighbors(p[0],p[1],h,w))
                        return (-deg,-near,abs(p[0]-h/2)+abs(p[1]-w/2),p)
                    r,c=min(safe,key=sk)
                else:
                    rates=global_rates(hidden,eqs,int(obs.get("mines",0) or 0),self.known)
                    candidates=[p for p in hidden if grid[p[0]][p[1]]=="unknown"]
                    if not candidates:return
                    def rk(p):
                        risk=rates.get(p,1.0); exposed=sum(q in revealed for q in neighbors(p[0],p[1],h,w)); deg=sum(q in hidden for q in neighbors(p[0],p[1],h,w))
                        return (risk,-exposed,-deg,abs(p[0]-h/2)+abs(p[1]-w/2),p)
                    r,c=min(candidates,key=rk)
                action={"type":"reveal","row":int(r),"col":int(c)}
            try:
                nxt=ctx.act(action); obs=nxt if isinstance(nxt,dict) else game.observe()
            except Exception:
                try:obs=game.observe()
                except Exception:return

def build(ctx):return Harness(ctx)
