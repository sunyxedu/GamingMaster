"""Conservative range/equity harness for mc-pot-odds-v2."""
from __future__ import annotations
import hashlib, itertools, json, math, random, re
try:
    from metaharness.core.contract import BudgetExceeded
except Exception:
    class BudgetExceeded(Exception): pass

RANKS = "23456789TJQKA"; SUITS = "cdhs"
DECK = tuple(range(52))
def cv(s): return RANKS.index(s[0])*4 + SUITS.index(s[1])
def rv(c): return c//4 + 2
def smask(cards):
    m=0
    for c in cards: m |= 1 << (rv(c)-2)
    return m
def straight(m):
    for hi in range(14,4,-1):
        x=0
        for r in range(hi-4,hi+1): x |= 1 << (r-2)
        if m & x == x: return hi
    return 5 if m & 0x100F == 0x100F else 0
def rank(cards):
    cards=list(cards); cnt=[0]*15; suits=[0]*4; m=0
    for c in cards:
        r=rv(c); cnt[r]+=1; suits[c%4] |= 1 << (r-2); m |= 1 << (r-2)
    flush=next((x for x in suits if x.bit_count()>=5),0)
    if flush and straight(flush): return (8,straight(flush))
    q=[r for r in range(14,1,-1) if cnt[r]>=4]
    t=[r for r in range(14,1,-1) if cnt[r]>=3]
    p=[r for r in range(14,1,-1) if cnt[r]>=2]
    if q: return (7,q[0],max(r for r in range(14,1,-1) if r!=q[0] and cnt[r]))
    if t and len(p)>=2: return (6,t[0],next(x for x in p if x!=t[0]))
    if flush: return (5,)+tuple(r+2 for r in range(12,-1,-1) if flush&(1<<r))[:5]
    st=straight(m)
    if st: return (4,st)
    if t: return (3,t[0])+tuple(r for r in range(14,1,-1) if r!=t[0] and cnt[r])[:2]
    if len(p)>=2: return (2,p[0],p[1],max(r for r in range(14,1,-1) if r not in p[:2] and cnt[r]))
    if p: return (1,p[0])+tuple(r for r in range(14,1,-1) if r!=p[0] and cnt[r])[:3]
    return (0,)+tuple(r for r in range(14,1,-1) if cnt[r])[:5]
def cmp(a,b): return 1.0 if a>b else .5 if a==b else 0.0
def legal(o,a):
    ids={x.get('id') for x in o.get('candidates',[])}
    if isinstance(a,dict):
        a=("raise_to "+str(a.get('amount'))) if a.get('action')=='raise_to' else a.get('action')
    if not isinstance(a,str): return None
    a=a.strip()
    if a in ids: return a
    m=re.fullmatch(r'raise_to (\d+)',a); l=o.get('legal',{})
    if m and l.get('can_raise') and l.get('min_raise_to')<=int(m[1])<=l.get('max_raise_to'): return a
    return None
def fallback(o):
    ids=[x['id'] for x in o.get('candidates',[])]
    for x in ('check','call','fold'):
        if x in ids:return x
    return ids[0] if ids else 'check'
def potfrac(o,f):
    l=o['legal']; b=max(o.get('street_bets',[0,0])); n=b+int(f*(o.get('pot',0)+o.get('to_call',0)))
    if not l.get('can_raise'): return fallback(o)
    n=max(l['min_raise_to'],min(l['max_raise_to'],n)); return 'raise_to '+str(n)

class Harness:
    def __init__(self,ctx): self.ctx=ctx; self.hand=None; self.hole=(); self.cache={}; self.rng=random.Random(91421); self.last_illegal=0
    def stop(self): return self.ctx.done or self.ctx.budget.remaining().get('wall_s',0)<8
    def eq(self,o):
        h=tuple(cv(x) for x in o.get('hole_cards',[])); b=tuple(cv(x) for x in o.get('board',[])); key=(h,b)
        if key in self.cache:return self.cache[key]
        if not b: # deliberately conservative heads-up preflop prior
            a,c=sorted((rv(x) for x in h),reverse=True); q=.25+.018*(a-2)+.012*(c-2)
            if a==c:q+=.18
            if h[0]%4==h[1]%4:q+=.035
            if a-c<=1:q+=.025
            q=min(.82,max(.08,q)); self.cache[key]=q; return q
        used=set(h+b); avail=[x for x in DECK if x not in used]; n=80 if len(b)<5 else 1; w=0.
        seed=int.from_bytes(hashlib.blake2s(bytes(h+b),digest_size=8).digest(),'big'); r=random.Random(seed)
        for _ in range(n):
            if self.stop(): raise BudgetExceeded()
            if len(b)<5: run=b+tuple(r.sample([x for x in avail],5-len(b)))
            else: run=b
            rem=[x for x in avail if x not in run[len(b):]]
            vil=tuple(r.sample(rem,2)); w+=cmp(rank(h+run),rank(vil+run))
        q=w/n; self.cache[key]=q; return q
    def action(self,o):
        if self.hand!=o.get('hand_id'):
            self.hand=o.get('hand_id'); self.hole=tuple(cv(x) for x in o.get('hole_cards',[])); self.cache={}
        st=o.get('street'); h=self.hole; q=self.eq(o); to=o.get('to_call',0); p=max(1,o.get('pot',0)); l=o.get('legal',{}); hero=o.get('hero_seat')
        if st=='preflop':
            a,b=sorted((rv(x) for x in h),reverse=True); suited=h[0]%4==h[1]%4; strong=(a==b or a>=13 or (a>=11 and b>=8) or suited and a>=9)
            if hero==1 and not o.get('history'):
                return potfrac(o,.75) if strong or suited or a>=10 else fallback(o)
            if to==0:return potfrac(o,.75) if strong and q>.57 else fallback(o)
            price=to/(p+to)
            return 'call' if q>price+.06 and (strong or suited) else fallback(o)
        board=[cv(x) for x in o.get('board',[])]; cat=rank(h+tuple(board))[0] if board else 0
        # Critical structural rule: never infer a private nut hand from a four-flush
        # or four-straight board. River raises require an actually robust made hand.
        if len(board)==5 and to>0:
            board_flush=max(sum(1 for c in board if c%4==s) for s in range(4))>=4
            board_straight=straight(smask(board))>0
            danger=board_flush or board_straight or len({rv(c) for c in board})<5
            need=to/(p+to)+(.10 if danger else .03)
            if q>=need and q>=.58:return 'call'
            return fallback(o)
        if to>0:
            need=to/(p+to)+(.06 if len(board)<5 else .03)
            return 'call' if q>=need else fallback(o)
        if q>=.76 or (q>=.63 and cat>=2): return potfrac(o,.55 if len(board)==3 else .75)
        return fallback(o)
    def run_episode(self,game):
        o=self.ctx.observe()
        while not self.ctx.done and not o.get('done',False) and not self.stop():
            try:
                a=legal(o,self.action(o)) or fallback(o); old=o
                o=self.ctx.act(a) or self.ctx.observe(); pr=self.ctx.progress()
                bad=max(pr.get('illegal_actions',0),o.get('illegal_actions',0))>self.last_illegal or bool(o.get('action_error'))
                self.last_illegal=max(pr.get('illegal_actions',0),o.get('illegal_actions',0))
                if bad:self.ctx.log('policy_recovery',action=a,progress=pr,error=o.get('action_error'))
                self.ctx.log('policy_step',action=a,progress=pr)
            except (BudgetExceeded,TimeoutError,RuntimeError):
                if self.ctx.done: return
                try:
                    o=self.ctx.observe(); o=self.ctx.act(fallback(o)) or self.ctx.observe()
                except Exception:return
            except Exception:
                if self.ctx.done:return
                try:o=self.ctx.act(fallback(self.ctx.observe())) or self.ctx.observe()
                except Exception:return

def build(ctx): return Harness(ctx)
