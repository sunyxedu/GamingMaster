"""Wide-button pressure with a compact equity-aware response policy."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import re
from metaharness.core.contract import BudgetExceeded

RANKS='23456789TJQKA'; SUITS='cdhs'; DECK=tuple(range(52))
def card(s): return RANKS.index(s[0])*4+SUITS.index(s[1])
def rank(cards):
    cnt=[0]*15; suits=[0]*4; mask=0
    for c in cards:
        r=c//4+2; cnt[r]+=1; suits[c%4]|=1<<(r-2); mask|=1<<(r-2)
    fm=next((x for x in suits if x.bit_count()>=5),0)
    def straight(m):
        for hi in range(12,3,-1):
            if m & (31<<(hi-4)) == 31<<(hi-4): return hi+2
        return 5 if m&0x100f==0x100f else 0
    if fm and straight(fm): return (8,straight(fm))
    qs=[r for r in range(14,1,-1) if cnt[r]==4]
    ts=[r for r in range(14,1,-1) if cnt[r]>=3]
    ps=[r for r in range(14,1,-1) if cnt[r]>=2]
    if qs:return (7,qs[0],max(r for r in range(2,15) if cnt[r] and r!=qs[0]))
    if ts and len(ps)>=2:return (6,ts[0],next(r for r in ps if r!=ts[0]))
    if fm:return (5,)+tuple(r+2 for r in range(12,-1,-1) if fm&(1<<r))[:5]
    st=straight(mask)
    if st:return (4,st)
    if ts:return (3,ts[0])+tuple(r for r in range(14,1,-1) if cnt[r] and r!=ts[0])[:2]
    if len(ps)>=2:return (2,ps[0],ps[1],max(r for r in range(2,15) if cnt[r] and r not in ps[:2]))
    if ps:return (1,ps[0])+tuple(r for r in range(14,1,-1) if cnt[r] and r!=ps[0])[:3]
    return (0,)+tuple(r for r in range(14,1,-1) if cnt[r])[:5]
def win(a,b):return 1. if a>b else .5 if a==b else 0.
def fallback(o):
    ids=[x.get('id') for x in o.get('candidates',[]) if isinstance(x,dict)]
    for x in ('check','fold','call'):
        if x in ids:return x
    return ids[0] if ids else 'check'
def legal(o,a):
    if isinstance(a,dict): a=('raise_to '+str(a.get('amount')) if a.get('action')=='raise_to' else a.get('action'))
    if not isinstance(a,str):return None
    a=a.strip(); ids=[x.get('id') for x in o.get('candidates',[]) if isinstance(x,dict)]
    if a in ids:return a
    m=re.fullmatch(r'raise_to (\d+)',a); l=o.get('legal',{})
    if m and l.get('can_raise') and l.get('min_raise_to') is not None and l.get('max_raise_to') is not None:
        n=int(m[1])
        if l['min_raise_to']<=n<=l['max_raise_to']:return a
    return None
class Harness:
 def __init__(self,ctx):
    self.ctx=ctx; self.last_illegal=0; self.audit_calls=0; self.pending=None; self.style='wide'; self.seen_results=0
 def equity(self,o):
    h=tuple(card(x) for x in o['hole_cards']); b=tuple(card(x) for x in o.get('board',[]))
    if len(b)==5:
        fixed=rank(h+b); return sum(win(fixed,rank(x+b)) for x in itertools.combinations([c for c in DECK if c not in h+b],2))/1081
    key=bytes(h+b); seed=int.from_bytes(hashlib.blake2s(key,digest_size=8).digest(),'big')
    # Deterministic visible-card sampling; independent of the game deal and opponent RNG.
    import random
    rng=random.Random(seed); rem=[c for c in DECK if c not in h+b]; n=52 if len(b)==3 else 72; total=0.
    for i in range(n):
        if i%16==0 and (self.ctx.done or self.ctx.budget.remaining().get('wall_s',0)<5): raise BudgetExceeded('wall')
        draw=rng.sample(rem,2+5-len(b)); opp=draw[:2]; board=b+tuple(draw[2:]); total+=win(rank(h+board),rank(opp+board))
    return total/n
 def preflop(self,o):
    h=[card(x) for x in o['hole_cards']]; ranks=sorted((x//4+2 for x in h),reverse=True); a,b=ranks
    suited=h[0]%4==h[1]%4; gap=a-b
    pair=a==b; premium=pair and a>=12 or a==14 and b>=13
    hero=o['hero_seat']; price=o.get('to_call',0); hist=o.get('history',[]); opp_raise=any(x.get('seat')!=hero and x.get('action')=='raise_to' for x in hist)
    legalx=o.get('legal',{}); bets=o.get('street_bets',[100,50]); top=max(bets)
    def raise_to(n):
        if not legalx.get('can_raise'):return 'call' if price else 'check'
        # choose closest supplied candidate at/above target, keep normal opening exact
        target=max(legalx['min_raise_to'],min(legalx['max_raise_to'],int(n)))
        return 'raise_to '+str(target)
    if opp_raise:
        if premium:return raise_to(min(legalx.get('max_raise_to',top),top*3))
        # Opponent's preflop reraises are premium-only by specification.
        if pair and a>=10 or a==14 and b>=12 and suited:return 'call'
        return 'fold'
    if hero==1 and not hist:
        # Deliberately widen button opening: every pair, ace, king, suited hand,
        # connected/gapped broadway, and selected connected low cards.
        playable=(pair or a>=14 or b>=12 or suited or (gap<=2 and a>=6) or (a>=10 and b>=7))
        if not playable:return 'fold'
        return raise_to(250)
    if price==0:
        # BB checks freely; isolate limps with broad value, otherwise retain pot control.
        if a>=13 and b>=9 or pair and a>=9 or suited and a>=12:return raise_to(400)
        return 'check'
    if top<=250:
        # Wide button range and cheap price: defend with real equity/playability.
        if pair or a>=13 or suited or (gap<=1 and a>=8):return 'call'
        return 'fold'
    if premium:return raise_to(top*3)
    if pair and a>=10 or a>=14 and b>=12 and suited:return 'call'
    return 'fold'
 def postflop(self,o):
    eq=self.equity(o); p=max(1,float(o.get('pot',1))); c=float(o.get('to_call',0)); board=o.get('board',[]); street=len(board)
    legalx=o.get('legal',{}); hero=o.get('hero_seat',0); bets=o.get('street_bets',[0,0]); top=max(bets)
    # Move the threshold up OOP on unfinished streets to account for poor realization.
    realization=(.045 if hero==0 and street<5 else .02 if street<5 else 0)
    if c>0:
        odds=c/(p+c); threshold=odds+realization+.035
        if eq<threshold:return 'fold'
        # Raise only strong equity; choose a modest legal price and avoid thin stacks.
        if legalx.get('can_raise') and eq>=.88:
            target=min(legalx['max_raise_to'], max(legalx['min_raise_to'], int(top+.65*(p+c))))
            return 'raise_to '+str(target)
        return 'call'
    # Bet for value against the opponent's explicit .62/.85 equity thresholds.
    if legalx.get('can_raise') and eq>=.64:
        target=max(legalx['min_raise_to'],min(legalx['max_raise_to'],int(top+.68*p)))
        return 'raise_to '+str(target)
    # A low-frequency bluff only when board is dry/high-card and equity has blockers.
    if legalx.get('can_raise') and eq<.32 and street>=3:
        h=tuple(card(x) for x in o['hole_cards']); b=tuple(card(x) for x in board)
        z=int.from_bytes(hashlib.blake2s(bytes(h+b),digest_size=4).digest(),'big')%10
        if z==0:
            target=max(legalx['min_raise_to'],min(legalx['max_raise_to'],int(top+.45*p)))
            return 'raise_to '+str(target)
    return 'check'
 def audit(self,o,why):
    if self.audit_calls>=2 or self.ctx.done:return
    rem=self.ctx.budget.remaining()
    if rem.get('system2_calls',0)<=0 or rem.get('wall_s',0)<160:return
    self.audit_calls+=1
    prompt=('We play heads-up no-limit Holdem against the fixed mc-pot-odds-v2 policy. This harness tests a wide small-blind opening range (pair, ace, king, suited, connected) because the opponent BB calls by equity threshold, and uses sampled equity after the flop. Review evidence and suggest only a reusable posture: wide, standard, or cautious. Do not provide an action. Reply JSON {"style":"wide|standard|cautious","lesson":"short"}. Evidence: '+json.dumps({'why':why,'completed_hands':o.get('completed_hands'),'recent':o.get('public_stats'),'pending':self.pending,'obs':{k:o.get(k) for k in ('street','hole_cards','board','to_call','pot','history')}}))
    try:
        r=self.ctx.think(prompt,timeout_s=130.)
        txt=r.text or ''; i=txt.find('{'); j=txt.rfind('}')
        v=json.loads(txt[i:j+1]) if i>=0 and j>i else {}
        if v.get('style') in ('wide','standard','cautious'):self.style=v['style']
        self.ctx.log('sparse_review',reason=why,style=self.style,accepted=bool(v))
    except Exception as e:self.ctx.log('review_unavailable',error=type(e).__name__)
 def run_episode(self,game):
    ctx=self.ctx; o=ctx.observe(); reviewed=False
    while not ctx.done and not o.get('done',False):
        try:
            if not reviewed and o.get('completed_hands',0)>=1:
                self.audit(o,'first settled hand; check whether wide entry is producing profitable realization'); reviewed=True
            if self.pending and self.audit_calls<2:
                self.audit(o,'execution/plan failure; re-evaluate the policy'); self.pending=None
            action=self.preflop(o) if o.get('street')=='preflop' else self.postflop(o)
            action=legal(o,action) or fallback(o)
            old=o; prog_before=ctx.progress(); illegal_before=max(self.last_illegal,prog_before.get('illegal_actions',0))
            o=ctx.act(action)
            if o is None:o=ctx.observe()
            prog=ctx.progress(); illegal=max(o.get('illegal_actions',0),prog.get('illegal_actions',0))
            if o.get('action_error') or illegal>illegal_before or prog.get('since_obs_change',0)>=2:
                self.pending={'action':action,'progress':prog,'error':o.get('action_error')}; self.last_illegal=illegal
            else:self.pending=None; self.last_illegal=illegal
            ctx.log('policy_step',action=action,equity_note='sampled' if o.get('street')!='preflop' else 'preflop class',progress=prog)
        except (BudgetExceeded,TimeoutError):return
        except Exception as e:
            if ctx.done:return
            self.pending={'error':type(e).__name__}; ctx.log('recovery',error=type(e).__name__)
            try:
                o=ctx.observe(); o=ctx.act(fallback(o)) or ctx.observe(); ctx.progress()
            except Exception:return

def build(ctx):return Harness(ctx)
