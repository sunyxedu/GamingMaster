"""Deadline-aware Hanabi harness.

This candidate keeps the public discard channel from 0008, but adds an explicit
turn-access planner: once the deck is short (or the final countdown starts),
actions are scored by the number of prerequisite links they can still execute
before the last available turn.  All inference remains seat-local and every
posted action is selected from the current legal menu.
"""
from __future__ import annotations

# The implementation is filled below from the conservative exact-belief policy;
# keeping it in one module makes the pre-flight contract self-contained.
import json, re
from collections import Counter

COLORS = 'RYGWB'
TYPES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = {x: (3 if x[1] == 1 else 1 if x[1] == 5 else 2) for x in TYPES}

def subsets(mask):
    while True:
        yield mask
        if not mask: break
        mask = (mask - 1) & mask

def marginals(domains, counts):
    n = len(domains); full = (1 << n) - 1
    choices = []
    for typ in TYPES:
        allowed = sum(1 << i for i,d in enumerate(domains) if typ in d)
        cnt = max(0, counts.get(typ, 0)); opts=[]; sub=allowed
        while True:
            k = sub.bit_count()
            if k <= cnt:
                w=1
                for j in range(k): w *= cnt-j
                opts.append((sub,w))
            if not sub: break
            sub=(sub-1)&allowed
        choices.append(opts)
    def extend(prev, opts):
        nxt=[0]*(full+1)
        for used,v in enumerate(prev):
            if v:
                for sub,w in opts:
                    if not (used & sub): nxt[used|sub] += v*w
        return nxt
    f=[[1]+[0]*full]
    for opts in choices: f.append(extend(f[-1],opts))
    b=[None]*(len(TYPES)+1); b[-1]=[1]+[0]*full
    for i in range(len(TYPES)-1,-1,-1): b[i]=extend(b[i+1],choices[i])
    total=f[-1][full]; out=[{} for _ in domains]
    if not total:
        for i,d in enumerate(domains):
            den=sum(max(0,counts.get(x,0)) for x in d)
            out[i]={x:max(0,counts.get(x,0))/den for x in d if den and counts.get(x,0)>0}
        return out
    for t,typ in enumerate(TYPES):
        for sub,w in choices[t]:
            if not sub: continue
            rest=full^sub; ways=0
            for left in subsets(rest): ways += f[t][left]*b[t+1][rest^left]
            if ways:
                for i in range(n):
                    if sub&(1<<i): out[i][typ]=out[i].get(typ,0)+ways*w/total
    return out

def remaining(obs):
    c=dict(COPIES)
    for col in COLORS:
        for r in range(1,obs['fireworks'][col]+1): c[col,r]-=1
    for card in obs['discard_pile']: c[card['color'],card['rank']]-=1
    return c

def ceilings(obs,c):
    out={}
    for col in COLORS:
        out[col]=5
        for r in range(obs['fireworks'][col]+1,6):
            if c[col,r]<=0: out[col]=r-1; break
    return out

def literal(card): return {(c,r) for c in card['possible_colors'] for r in card['possible_ranks']}

class Policy:
    def __init__(self,ctx):
        self.ctx=ctx; self.promises={}; self.sessions={}; self.calls=0; self.last_call=-9
    def domain(self,card,counts):
        d={x for x in literal(card) if counts.get(x,0)>0}; p=self.promises.get(card['card_id'])
        if p is not None and d&p: d &= p
        return d
    def belief(self,obs,seat,private=True):
        counts=remaining(obs)
        if private:
            for s,h in enumerate(obs['hands']):
                if s!=seat:
                    for card in h:
                        if card['color'] is not None: counts[card['color'],card['rank']]-=1
        dom=[self.domain(card,counts) for card in obs['hands'][seat]]
        return marginals(dom,counts),dom
    def playable(self,x,obs): return x[1]==obs['fireworks'][x[0]]+1
    def dead(self,x,obs,cap): return x[1]<=obs['fireworks'][x[0]] or x[1]>cap[x[0]]
    def facts(self,obs,seat,private=True):
        counts=remaining(obs); cap=ceilings(obs,counts); probs,dom=self.belief(obs,seat,private)
        vis=Counter()
        if private:
            for s,h in enumerate(obs['hands']):
                if s!=seat:
                    for card in h:
                        if card['color'] is not None: vis[card['color'],card['rank']]+=1
        out=[]
        for pr,d in zip(probs,dom):
            pp=pd=pc=loss=0.0
            for x,p in pr.items():
                isdead=self.dead(x,obs,cap); pp+=p*self.playable(x,obs); pd+=p*isdead
                crit=counts[x]==1 and not isdead; pc+=p*crit
                if not isdead:
                    dist=x[1]-obs['fireworks'][x[0]]
                    loss += p*((cap[x[0]]-x[1]+1)*(0.9**(dist-1)) if crit else (0.42/counts[x]+0.22/max(1,dist))*(0.25 if vis[x] else 1))
            out.append({'index':len(out),'p_play':pp,'p_dead':pd,'p_critical':pc,'loss':loss,'domain':d,'prob':pr})
        return out
    def protected(self,card): return card.get('color_clued') or card.get('rank_clued') or card['card_id'] in self.promises
    def clue_effect(self,obs,a,touched_ids=None):
        hand=obs['hands'][a['target']]; attr='color' if a['type']=='REVEAL_COLOR' else 'rank'; val=a[attr]; counts=remaining(obs)
        touched=[i for i,c in enumerate(hand) if ((c[attr]==val) if touched_ids is None else c['card_id'] in touched_ids)]
        k=0 if attr=='color' else 1; ds=[]
        for i,card in enumerate(hand): ds.append({x for x in self.domain(card,counts) if (x[k]==val)==(i in touched)})
        focus=promise=None
        if attr=='color':
            typ=(val,obs['fireworks'][val]+1); elig=[i for i in touched if typ in ds[i]]
            if elig:
                focus=next((i for i in elig if not hand[i].get('color_clued')),elig[0]); promise={typ}; ds[focus]&=promise
        return touched,ds,focus,promise
    def discard_codebook(self,obs):
        me=obs['observer']; other=1-me; own=self.facts(obs,me,False); partner=self.facts(obs,other,False); table={i:None for i in range(len(own))}
        if not own or not partner or obs.get('final_turns_remaining')==1 or any(f['p_play']>1-1e-10 for f in partner): return table
        elig=[f['index'] for f in partner if f['p_play']>1e-10]
        if not elig:return table
        h=obs['hands'][me]; ordered=sorted(own,key=lambda f:(f['loss']+0.45*(self.protected(h[f['index']]) and f['p_dead']<1-1e-10)-0.2*(f['p_dead']>1-1e-10),f['index']))
        for j,f in enumerate(ordered[1:]): table[f['index']]=elig[(obs['turn']//2+j)%len(elig)]
        return table
    def public_view(self,obs):
        keys=('observer','turn','fireworks','information_tokens','life_tokens','deck_size','discard_pile','num_players')
        return {**{k:obs[k] for k in keys},'final_turns_remaining':obs.get('final_turns_remaining'),'hands':[[{**c,'color':None,'rank':None} for c in h] for h in obs['hands']]}
    def accept(self,before,after):
        ev=(after.get('history') or [])[-1:]
        if not ev or ev[0].get('turn')!=before.get('turn'): return
        e=ev[0]
        if e['type'].startswith('REVEAL_'):
            _,_,focus,p=self.clue_effect(before,e,set(e.get('touched_card_ids',[])))
            if focus is not None: self.promises[before['hands'][e['target']][focus]['card_id']]=p
        elif e['type']=='DISCARD':
            slot=self.discard_codebook(before).get(e['card_index'])
            if slot is not None:
                target=1-before['observer']; card=before['hands'][target][slot]
                p={x for x in self.domain(card,remaining(before)) if self.playable(x,before)}
                if p:self.promises[card['card_id']]=p
    def access(self,obs,face):
        """How many prerequisite links remain executable before deck/end countdown."""
        final=obs.get('final_turns_remaining'); horizon=(final if final is not None else max(2,obs['deck_size']//2+2))
        if not face:return 0
        c,r=face; h=obs['fireworks'][c]; return max(0,min(5-r+1,horizon)) if r>h else 0
    def candidates(self,obs):
        me=obs['observer']; other=1-me; counts=remaining(obs); cap=ceilings(obs,counts); own=self.facts(obs,me); partner=self.facts(obs,other,False); hand=obs['hands'][me]; ph=obs['hands'][other]; final=obs.get('final_turns_remaining'); table=self.discard_codebook(obs); predicted=min(partner,key=lambda f:f['loss']+0.012*f['index'])['index'] if partner else None
        safe=[f['index'] for f in own if f['p_play']>1-1e-10]; psafe=[f['index'] for f in partner if f['p_play']>1-1e-10]; out=[]
        # Number of future actions available to each seat, used to penalize clues
        # that cannot be consumed before the end.
        horizon=final if final is not None else max(1,obs['deck_size']+2)
        for a in obs['legal_actions']:
            typ=a['type']; val=-99; info={}
            if typ=='PLAY':
                f=own[a['card_index']]; p=f['p_play']
                if p>1-1e-10:
                    unlock=sum(q*self.access(obs,x) for x,q in f['prob'].items()); val=2.8+0.35*unlock
                    info={'reason':'certain play','access_gain':round(unlock,2)}
                elif (final is not None or obs['deck_size']<=3) and obs['life_tokens']>1 and p>=.35:
                    val=1.2*p-(1-p)*(.25+f['loss']); info={'reason':'late risk','p_success':round(p,3)}
                else: continue
            elif typ=='DISCARD':
                f=own[a['card_index']]; sig=table.get(a['card_index'])
                if sig is not None and not self.playable((ph[sig]['color'],ph[sig]['rank']),obs): continue
                val=.95-1.7*f['loss']-.012*a['card_index']-(.45 if self.protected(hand[a['card_index']]) and f['p_dead']<1-1e-10 else 0)
                if obs['information_tokens']<=1:val+=.45
                if f['p_dead']>1-1e-10:val+=.35
                if final is not None:val=-.15
                if sig is not None: val += 2.5 + 0.15*self.access(obs,(ph[sig]['color'],ph[sig]['rank'])); info={'reason':'discard play signal','target_slot':sig}
            elif typ.startswith('REVEAL_'):
                touched,after,focus,promise=self.clue_effect(obs,a)
                if focus is not None and (ph[focus]['color'],ph[focus]['rank']) not in promise: continue
                enabled=[i for i,d in enumerate(after) if d and all(self.playable(x,obs) for x in d)]; newly=[i for i in enabled if i not in psafe]
                protect=0
                for i in touched:
                    x=(ph[i]['color'],ph[i]['rank'])
                    if counts[x]==1 and not self.dead(x,obs,cap) and i==predicted: protect=max(protect,1.5)
                # Clues are valuable only if the recipient has time to consume them.
                val=-.55+2.5*min(1,len(newly))+.65*max(0,len(newly)-1)+protect
                if final is not None: val=.9 if final>=2 and newly else -.4
                if horizon<=2 and not newly: val-=.8
                info={'reason':'play/save clue','new_safe':newly,'focus':focus}
            else: continue
            out.append({'action':a,'value':round(val,5),**info})
        out.sort(key=lambda x:x['value'],reverse=True); return out,own,safe
    def ask(self,obs,cands,facts):
        rem=self.ctx.budget.remaining(); seat=obs['observer']
        if self.calls>=2 or rem['system2_calls']<=2 or rem['wall_s']<900 or obs['turn']-self.last_call<6 or len(cands)<2 or cands[0]['value']-cands[1]['value']>1.0:return None
        opts=cands[:5]; state={'seat':seat,'turn':obs['turn'],'fireworks':obs['fireworks'],'clues':obs['information_tokens'],'lives':obs['life_tokens'],'deck':obs['deck_size'],'final':obs.get('final_turns_remaining'),'own':[{'slot':i,'id':c['card_id'],'p_play':round(facts[i]['p_play'],3),'p_dead':round(facts[i]['p_dead'],3),'loss':round(facts[i]['loss'],3)} for i,c in enumerate(obs['hands'][seat])],'partner':obs['hands'][1-seat],'options':[{'id':i,**x} for i,x in enumerate(opts)]}
        prompt=('Choose one listed legal Hanabi action. Maximize terminal score; never invent card faces or conventions. '
                'The deterministic planner values actions by future turn access and preserves critical cards. '
                'Reply only JSON {"choice": integer}.\n'+json.dumps(state,separators=(',',':')))
        self.calls+=1; self.last_call=obs['turn']
        try:
            fut=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=self.sessions.get(seat),timeout_s=min(90,rem['wall_s']-600)))
            res=fut.result(timeout=120); self.sessions[seat]=res.session_id
            m=re.search(r'\{[^{}]*"choice"\s*:\s*(-?\d+)[^{}]*\}',res.text)
            if m:
                i=int(m.group(1)); return opts[i] if 0<=i<len(opts) else None
        except Exception as exc:
            self.calls=2; self.ctx.log('planner_fallback',error=type(exc).__name__)
        return None
    def run_episode(self,game):
        self.promises={}; self.sessions={}; self.calls=0; self.last_call=-9; obs=self.ctx.observe(); failures=0
        for _ in range(220):
            if self.ctx.done or obs.get('done') or not obs.get('legal_actions'):break
            cands,facts,safe=self.candidates(obs); selected=(cands[0] if cands else {'action':obs['legal_actions'][0]})
            if cands and not safe: selected=self.ask(obs,cands,facts) or selected
            before=self.public_view(obs)
            try:new=self.ctx.act(selected['action'])
            except Exception as exc:
                if self.ctx.done:break
                failures+=1; self.ctx.log('action_failure',error=type(exc).__name__); obs=self.ctx.observe()
                if failures>=3:break
                continue
            obs=new if isinstance(new,dict) else self.ctx.observe()
            self.ctx.log('decision', seat=before.get('observer'), turn=before.get('turn'),
                         action=selected.get('action'), reason=selected.get('reason'),
                         value=selected.get('value'), deck=before.get('deck_size'),
                         final_turns=before.get('final_turns_remaining'))
            if obs.get('turn',-1)>before.get('turn',-1):
                self.accept(before,obs); ids={c['card_id'] for h in obs['hands'] for c in h}; self.promises={i:p for i,p in self.promises.items() if i in ids}; failures=0
            else: failures+=1
            if failures>=3:break

def build(ctx): return Policy(ctx)
