"""Canonical clue factors with bounded exact joint support deduction.

One sender-visible hand chooses one canonical truthful clue. The receiver
propagates constraints on hands compatible with that same choice.
Private exhaustive support closure removes impossible identities. The parent
prior then supplies approximate probabilities on those supports, rather than
treating partial policy conditioning as a calibrated behavioral posterior.
Only public clue factors cross seats; model sessions remain seat-local.
"""
from __future__ import annotations
import json
import re
import time
from collections import Counter

COLORS = 'RYGWB'
TYPES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = {x: (3 if x[1] == 1 else 1 if x[1] == 5 else 2) for x in TYPES}
FACE_INDEX = {x: i for i, x in enumerate(TYPES)}


def packet_factor(before, event):
    """Keep only public data needed to reproduce a historical encoder."""
    hand = before['hands'][event['target']]
    return dict(target=event['target'], ids=tuple(c['card_id'] for c in hand),
                hand_len=len(hand), turn=before['turn'], selected=symbol_of(event),
                touched=frozenset(event.get('touched_card_ids', [])),
                tables=utilities(before))


def joint_condition(domains, counts, ids, factors, revealed, *,
                    node_limit=180000, wall_limit=1.5, stop=None):
    """Exact weighted enumeration, or no result at all on budget exhaustion.

    The factors encode the *whole* deterministic clue choice, including the
    absence of every truthful command that would have beaten the chosen one.
    No prefix of the enumeration is used as an estimate or a certainty proof.
    """
    started = time.monotonic()
    n = len(ids)
    positions = {cid: i for i, cid in enumerate(ids)}
    options = [tuple(sorted(FACE_INDEX[x] for x in dom if counts.get(x, 0)>0))
               for dom in domains]
    if any(not d for d in options):
        return None, dict(status='empty_domain', nodes=0)
    compiled = []
    focus_use = Counter()
    for factor in factors:
        refs = []
        for cid in factor['ids']:
            if cid in positions:
                refs.append((positions[cid], None))
            elif cid in revealed:
                refs.append((-1, FACE_INDEX[revealed[cid]]))
            else:
                # Missing public reveal: do not approximate away an old latent
                # variable and then call the resulting probabilities exact.
                return None, dict(status='missing_reveal', nodes=0)
        k0 = factor['selected']
        focus = slot_for(k0, factor['turn'], factor.get('hand_len', len(refs)))
        fi, fixed = refs[focus]
        if fi >= 0:
            focus_use[fi] += 1
        table = factor['tables']
        selected_options = options[fi] if fi >= 0 else (fixed,)
        branches = {}
        for face_no in selected_options:
            value = table[k0//5][TYPES[face_no]]
            if value <= -999:
                branches[face_no] = None
                continue
            checks = []
            for k in range(10):
                if k == k0:
                    continue
                bad = sum(1 << i for i, face in enumerate(TYPES)
                          if (table[k//5][face], -k) > (value, -k0))
                if not bad:
                    continue
                attr = sum(1 << i for i, (c, r) in enumerate(TYPES)
                           if (r == k+1 if k < 5 else c == COLORS[k-5]))
                si, sf = refs[slot_for(k, factor['turn'], factor.get('hand_len', len(refs)))]
                if si < 0 and not (bad >> sf & 1):
                    continue
                # A clue is truthful iff at least one old hand slot matches.
                # Fixed public reveals remain available after hand shifts.
                always_present = any(j < 0 and (attr >> f & 1) for j, f in refs)
                attr_slots = tuple(j for j, f in refs if j >= 0 and
                                   any(attr >> x & 1 for x in options[j]))
                if always_present or attr_slots:
                    checks.append((si, sf, bad, attr, always_present, attr_slots))
            branches[face_no] = checks
        compiled.append((fi, fixed, branches))

    assigned = [-1] * n
    pool = [counts.get(x, 0) for x in TYPES]
    totals = [[0] * len(TYPES) for _ in range(n)]
    total_weight = 0
    nodes = 0
    worlds = 0
    order = sorted(range(n), key=lambda i: (len(options[i]), -focus_use[i], i))

    def compatible():
        for fi, fixed, branches in compiled:
            f = assigned[fi] if fi >= 0 else fixed
            if f < 0:
                continue
            checks = branches[f]
            if checks is None:
                return False
            for si, sf, bad, attr, present, slots in checks:
                s = assigned[si] if si >= 0 else sf
                if s < 0 or not (bad >> s & 1):
                    continue
                if present or any(assigned[j] >= 0 and (attr >> assigned[j] & 1)
                                  for j in slots):
                    return False
        return True

    class Incomplete(Exception):
        pass

    def visit(depth, weight):
        nonlocal nodes, total_weight, worlds
        nodes += 1
        if nodes > node_limit:
            raise Incomplete()
        if nodes % 256 == 0 and (time.monotonic()-started > wall_limit or
                                 (stop is not None and stop())):
            raise Incomplete()
        if not compatible():
            return
        if depth == n:
            worlds += 1
            total_weight += weight
            for i, f in enumerate(assigned):
                totals[i][f] += weight
            return
        i = order[depth]
        for f in options[i]:
            copies = pool[f]
            if copies:
                assigned[i] = f
                pool[f] -= 1
                visit(depth+1, weight*copies)
                pool[f] += 1
        assigned[i] = -1

    try:
        visit(0, 1)
    except Incomplete:
        return None, dict(status='budget_fallback', nodes=nodes,
                          elapsed=round(time.monotonic()-started, 4))
    if not total_weight:
        return None, dict(status='contradiction', nodes=nodes)
    probs = [{TYPES[f]: w/total_weight for f, w in enumerate(row) if w}
             for row in totals]
    return probs, dict(status='complete', nodes=nodes, worlds=worlds,
                      factors=len(factors), elapsed=round(time.monotonic()-started, 4))


def subsets(mask):
    sub = mask
    while True:
        yield sub
        if not sub:
            break
        sub = (sub - 1) & mask


def marginals(domains, counts):
    """Exact multiplicity-weighted matching of labeled slots, without replacement.

    Polynomial DP over slot subsets; no hidden world is used as a policy input.
    """
    n = len(domains)
    full = (1 << n) - 1
    choices = []
    for typ in TYPES:
        allowed = sum(1 << i for i, dom in enumerate(domains) if typ in dom)
        cnt = max(0, counts.get(typ, 0))
        opts = []
        for sub in subsets(allowed):
            k = sub.bit_count()
            if k <= cnt:
                weight = 1
                for j in range(k):
                    weight *= cnt - j
                opts.append((sub, weight))
        choices.append(opts)

    def extend(prev, opts):
        nxt = [0] * (full + 1)
        for used, val in enumerate(prev):
            if val:
                for sub, weight in opts:
                    if not (used & sub):
                        nxt[used | sub] += val * weight
        return nxt

    forward = [[1] + [0] * full]
    for opts in choices:
        forward.append(extend(forward[-1], opts))
    backward = [None] * (len(TYPES) + 1)
    backward[-1] = [1] + [0] * full
    for t in range(len(TYPES) - 1, -1, -1):
        backward[t] = extend(backward[t + 1], choices[t])
    total = forward[-1][full]
    answer = [dict() for _ in domains]
    if not total:
        # Conservative fallback if external observations contradict a promise.
        for i, dom in enumerate(domains):
            weights = {x: max(0, counts.get(x, 0)) for x in dom}
            den = sum(weights.values())
            answer[i] = {x: w / den for x, w in weights.items() if w} if den else {}
        return answer
    for t, typ in enumerate(TYPES):
        for sub, weight in choices[t]:
            if not sub:
                continue
            rest = full ^ sub
            ways = sum(forward[t][left] * backward[t + 1][rest ^ left]
                       for left in subsets(rest)) * weight
            if ways:
                for i in range(n):
                    if sub & (1 << i):
                        answer[i][typ] = answer[i].get(typ, 0.0) + ways / total
    return answer


def remaining(obs):
    counts = dict(COPIES)
    for c in COLORS:
        for r in range(1, obs['fireworks'][c] + 1):
            counts[c, r] -= 1
    for card in obs['discard_pile']:
        counts[card['color'], card['rank']] -= 1
    return counts


def ceilings(obs, counts):
    out = {}
    for c in COLORS:
        out[c] = 5
        for r in range(obs['fireworks'][c] + 1, 6):
            if counts[c, r] <= 0:
                out[c] = r - 1
                break
    return out


def literal(card):
    return {(c, r) for c in card['possible_colors'] for r in card['possible_ranks']}



def slot_for(symbol, turn, n):
    return (symbol % 5 + turn // 2) % n


def symbol_action(symbol, target):
    if symbol < 5:
        return dict(type='REVEAL_RANK', target=target, rank=symbol+1)
    return dict(type='REVEAL_COLOR', target=target, color=COLORS[symbol-5])


def symbol_of(action):
    return action['rank']-1 if action['type']=='REVEAL_RANK' else 5+COLORS.index(action['color'])


def loss_of(typ, obs, counts, caps):
    c,r=typ
    if r <= obs['fireworks'][c] or r > caps[c]:
        return 0.0
    distance=r-obs['fireworks'][c]
    potential=(caps[c]-r+1)*0.84**(distance-1)
    n=counts[typ]
    return potential if n <= 1 else potential*(0.15 if n==2 else 0.07)*(1.3 if obs['deck_size']<10 else 1.0)


def utilities(obs):
    """Every encoder quantity is public except the target slot face."""
    counts=remaining(obs); cap=ceilings(obs, counts)
    play={}; discard={}
    for face in TYPES:
        c,r=face
        playable=r==obs['fireworks'][c]+1
        play[face]=(6.0+0.35*(5-r)+0.5*(r==5)) if playable else -1000.0
        loss=loss_of(face,obs,counts,cap)
        discard[face]=2.2-3.5*loss+0.7*(loss==0)-1.5*playable
    return play,discard


def encode(obs, target):
    """Canonical message; cannot depend on sender's hand or private beliefs.

    Selecting the maximum with a fixed tie break makes every unchosen, more
    valuable legal message negative evidence about the recipient's whole hand.
    """
    hand=obs['hands'][target]
    if not hand:return None
    tables=utilities(obs)
    faces=[(c['color'],c['rank']) for c in hand]
    legal_symbols=[k for k in range(10) if any(
        (r==k+1 if k<5 else c==COLORS[k-5]) for c,r in faces)]
    values={k:tables[k//5][faces[slot_for(k,obs['turn'],len(hand))]] for k in legal_symbols}
    valid=[k for k in legal_symbols if values[k]>-999]
    # There is always a truthful color clue for a nonempty hand.
    k=max(valid,key=lambda j:(values[j],-j))
    return symbol_action(k,target), values[k]


class Policy:
    def __init__(self,ctx):
        self.ctx=ctx
        self.domains={}
        self.sessions={}
        self.calls=0
        self.advice=True
        self.last_call=-100
        self.factors=[]
        self.revealed={}

    def domain(self,card,counts):
        dom={x for x in literal(card) if counts.get(x,0)>0}
        # Public inverse-policy constraints, never a face copied from a hand.
        if card['card_id'] in self.domains:
            dom &= self.domains[card['card_id']]
        return dom

    def public_view(self,obs):
        return {**{k:obs[k] for k in ('observer','turn','fireworks','information_tokens',
                    'life_tokens','deck_size','discard_pile','num_players')},
                'hands':[[{**c,'color':None,'rank':None} for c in hand] for hand in obs['hands']],
                'final_turns_remaining':obs.get('final_turns_remaining')}

    def decode(self,before,event):
        """Sound inverse encoder via disjunctive branches and arc consistency.

        For every better competing message:
          (encoded slot is not better) OR (clue attribute absent from hand).
        Branch on the selected slot face, propagate each implication, then union
        feasible supports. This is an OVER-approximation, never sample certainty.
        Counts add singleton capacity propagation. Correlations not represented
        by the union are intentionally forgotten rather than asserted as facts.
        """
        target=event['target'];hand=before['hands'][target];n=len(hand)
        counts=remaining(before); tables=utilities(before)
        selected=symbol_of(event);focus=slot_for(selected,before['turn'],n)
        touched=set(event.get('touched_card_ids',[]))
        axis=1 if selected<5 else 0
        attr=selected+1 if selected<5 else COLORS[selected-5]
        ds=[]
        for c in hand:
            hit=c['card_id'] in touched
            ds.append({x for x in self.domain(c,counts) if (x[axis]==attr)==hit})
        literal_ds=[set(x) for x in ds]
        if selected<5:
            ds[focus]={x for x in ds[focus] if tables[0][x]>-999}
        supports=[set() for _ in hand]
        for face in sorted(ds[focus]):
            value=tables[selected//5][face]
            if value<=-999:continue
            branch=[set(x) for x in ds];branch[focus]={face}
            clauses=[]
            for k in range(10):
                if k==selected:continue
                s=slot_for(k,before['turn'],n)
                bad={x for x in TYPES if (tables[k//5][x],-k)>(value,-selected)}
                attr_set={x for x in TYPES if (x[1]==k+1 if k<5 else x[0]==COLORS[k-5])}
                clauses.append((s,bad,attr_set))
            valid=True
            for _ in range(32):
                old=tuple(frozenset(x) for x in branch)
                for s,bad,attrs in clauses:
                    branch[s]-=bad & attrs
                    if not branch[s]:valid=False;break
                    if branch[s] <= bad:
                        for j in range(n):branch[j]-=attrs
                    if any(d and d<=attrs for d in branch):
                        branch[s]-=bad
                    if any(not d for d in branch):valid=False;break
                if not valid:break
                fixed=Counter(next(iter(d)) for d in branch if len(d)==1)
                if any(v>counts[k] for k,v in fixed.items()):valid=False;break
                used={k for k,v in fixed.items() if v==counts[k]}
                for d in branch:
                    if len(d)>1:d-=used
                if any(not d for d in branch):valid=False;break
                if old==tuple(frozenset(x) for x in branch):break
            if valid:
                for out,d in zip(supports,branch):out.update(d)
        if any(not d for d in supports):
            # A contradiction disables EXTRA inference, preserving literal facts
            # and the universally verified rank play certificate.
            supports=ds if all(ds) else literal_ds
        return supports,focus,literal_ds

    def accept(self,before,after):
        events=after.get('history',[])[-1:]
        if not events or events[0].get('turn')!=before['turn']:return
        e=events[0]
        if e['type'].startswith('REVEAL_'):
            self.factors.append(packet_factor(before,e))
            domains,focus,literal_domains=self.decode(before,e)
            removed=sum(len(a)-len(b) for a,b in zip(literal_domains,domains))
            for c,d in zip(before['hands'][e['target']],domains):
                if d:self.domains[c['card_id']]=d
            self.ctx.log('inverse_packet',turn=before['turn'],target=e['target'],focus=focus,
                         extra_identities_removed=removed,
                         domains=[[c+str(r) for c,r in sorted(d)] for d in domains])
        elif e['type'] in ('PLAY','DISCARD'):
            self.revealed[e['card_id']]=(e['color'],e['rank'])
        active={c['card_id'] for h in after['hands'] for c in h}
        self.domains={i:d for i,d in self.domains.items() if i in active}
        self.factors=[f for f in self.factors if any(i in active for i in f['ids'])]
        needed={i for f in self.factors for i in f['ids']}
        self.revealed={i:x for i,x in self.revealed.items() if i in needed}

    def facts(self,obs,seat,private=False):
        counts=remaining(obs);cap=ceilings(obs,counts);pool=dict(counts)
        if private:
            for s,hand in enumerate(obs['hands']):
                if s!=seat:
                    for c in hand:pool[c['color'],c['rank']]-=1
        domains=[self.domain(c,pool) for c in obs['hands'][seat]]
        # Exact without-replacement weights GIVEN these unary support sets.
        # The full policy-conditioned posterior has additional correlations
        # which are dropped; probabilities are therefore an approximation.
        probs=marginals(domains,pool)
        if private:
            factors=[f for f in self.factors if f['target']==seat]
            if factors:
                exact,diagnostic=joint_condition(domains,pool,
                    [c['card_id'] for c in obs['hands'][seat]],factors,self.revealed,
                    stop=lambda: bool(getattr(self.ctx,'done',False)))
                if exact is not None:
                    diagnostic['exact_weight_delta']=round(max(
                        (abs(a.get(x,0)-b.get(x,0)) for a,b in zip(exact,probs)
                         for x in TYPES),default=0),6)
                    # Exhaustive support is a logical consequence of the
                    # protocol. Its relative weights still omit other policy
                    # selection effects. Keep the parent's prior, intersected
                    # with identities that have a complete compatible witness.
                    refined=[set(p) for p in exact]
                    diagnostic['private_identities_removed']=sum(
                        len(a-b) for a,b in zip(domains,refined))
                    projected=marginals(refined,pool)
                    diagnostic['max_marginal_change']=round(max(
                        (abs(a.get(x,0)-b.get(x,0)) for a,b in zip(projected,probs)
                         for x in TYPES),default=0),6)
                    domains=refined
                    probs=projected
                self.ctx.log('joint_packet_belief',seat=seat,turn=obs['turn'],**diagnostic)
        out=[]
        for i,(dom,prob) in enumerate(zip(domains,probs)):
            out.append(dict(index=i,domain=dom,prob=prob,
                play=sum(p for (c,r),p in prob.items() if r==obs['fireworks'][c]+1),
                dead=sum(p for (c,r),p in prob.items() if r<=obs['fireworks'][c] or r>cap[c]),
                loss=sum(p*loss_of(x,obs,counts,cap) for x,p in prob.items()),
                critical=sum(p for (c,r),p in prob.items() if counts[c,r]==1 and obs['fireworks'][c]<r<=cap[c])))
        return out

    def candidates(self,obs):
        seat=obs['observer'];target=(seat+1)%obs['num_players']
        own=self.facts(obs,seat,True);partner=self.facts(obs,target)
        tokens=obs['information_tokens'];final=obs.get('final_turns_remaining')
        entries=[];packet=encode(obs,target) if tokens else None
        incoming=obs.get('history',[])[-1:]
        suggested=None
        if incoming:
            e=incoming[0]
            if e.get('target')==seat and e.get('turn')==obs['turn']-1 and e['type']=='REVEAL_COLOR':
                suggested=slot_for(symbol_of(e),e['turn'],len(own))
        for action in obs['legal_actions']:
            kind=action['type'];value=-1000;why=''
            if kind=='PLAY':
                f=own[action['card_index']]
                if f['play']>1-1e-9:
                    value=10+0.3*f['critical'];why='certain play from literal/public inverse support'
                elif final is not None:
                    p=f['play'];score=sum(obs['fireworks'].values())
                    gain=p if obs['life_tokens']>1 else p-(1-p)*score
                    if gain>0:
                        value=2+gain if final==1 else gain
                        why='positive expected final-round play'
            elif kind=='DISCARD':
                f=own[action['card_index']]
                value=1.1+1.6*f['dead']-3.5*f['loss']+2*(tokens<=1)
                why='private posterior discard'
                if action['card_index']==suggested:
                    # Parent's known-face discard recommendation yields to both
                    # own scoring and a scoring message. It is not a promise of
                    # absolute deadness, and does not constrain future domains.
                    value=max(value,3.2)
                    why='canonical best visible discard recommendation'
                if final is not None:value=-1-f['loss']
            elif packet and action==packet[0]:
                k=symbol_of(action);focus=slot_for(k,obs['turn'],len(partner))
                value=packet[1];why='canonical whole-hand information packet'
                known=partner[focus]['play']>1-1e-9 if k<5 else partner[focus]['dead']>1-1e-9
                if known:value-=4.5
                if k>=5:
                    value-=2*(tokens<=1)
                if final is not None:
                    value=4 if k<5 and final>1 and not known else -2
            if value>-999:entries.append(dict(action=action,value=value,why=why))
        entries.sort(key=lambda x:x['value'],reverse=True)
        return entries,own

    def ask(self,obs,entries,own):
        budget=self.ctx.budget.remaining()
        if (not self.advice or self.calls>=2 or obs['turn']<16 or
            obs['turn']-self.last_call<12 or budget.get('wall_s',0)<180 or
            budget.get('system2_calls',0)<=1 or len(entries)<2 or
            entries[0]['value']>=6 or entries[0]['value']-entries[1]['value']>1):return None
        options=entries[:5];seat=obs['observer']
        state=dict(seat=seat,turn=obs['turn'],fireworks=obs['fireworks'],
            clues=obs['information_tokens'],lives=obs['life_tokens'],deck=obs['deck_size'],
            final_turns=obs.get('final_turns_remaining'),
            own=[dict(slot=f['index'],p_play=f['play'],p_dead=f['dead'],discard_loss=f['loss'],
                      possible=[c+str(r) for c,r in sorted(f['domain'])]) for f in own],
            visible_hands={str(i):h for i,h in enumerate(obs['hands']) if i!=seat},
            options=[dict(id=i,**e) for i,e in enumerate(options)])
        prompt=('Choose one listed action for two-player Hanabi, maximizing final score; three lost lives gives zero. '
                'You are only the indicated seat. The fixed encoder offers at most ONE canonical clue: rank clues '
                'promise a play at a rotating slot; color clues recommend the best encoded discard. The clue choice '
                'also excludes alternative hands where a better legal command exists. Code performs sound public '
                'constraint propagation and finite-population private beliefs. Do not invent messages or card faces. '
                'Preserve low prerequisites and unique future cards, but avoid stalling. Reply only JSON '
                '{"choice": integer}.\n'+json.dumps(state,separators=(',',':')))
        self.calls+=1;self.last_call=obs['turn']
        try:
            fut=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=self.sessions.get(seat),timeout_s=60))
            result=fut.result(timeout=70)
            self.sessions[seat]=result.session_id
            m=re.search(r'"choice"\s*:\s*(\d+)',result.text)
            if m and int(m.group(1))<len(options):return options[int(m.group(1))]
        except Exception as exc:
            self.advice=False
            self.ctx.log('adviser_fallback',error=type(exc).__name__,seat=seat)
        return None

    def run_episode(self,game):
        self.domains={};self.sessions={};self.calls=0;self.advice=True;self.last_call=-100
        self.factors=[];self.revealed={}
        obs=self.ctx.observe();failures=0
        for _ in range(240):
            if self.ctx.done or not isinstance(obs,dict) or obs.get('done'):return
            legal=obs.get('legal_actions',[])
            if not legal:return
            entries,own=self.candidates(obs)
            if not entries:return
            selected=self.ask(obs,entries,own) or entries[0]
            if self.ctx.done:return
            action=selected['action']
            if action not in legal:return
            before=self.public_view(obs)
            self.ctx.log('decision',seat=obs['observer'],turn=obs['turn'],selected=selected)
            try:new=self.ctx.act(action)
            except Exception as exc:
                if self.ctx.done:return
                self.ctx.log('act_failure',error=type(exc).__name__)
                failures+=1
                if failures>=2:return
                obs=self.ctx.observe();continue
            new=new if isinstance(new,dict) else self.ctx.observe()
            if new.get('turn',0)>obs['turn']:
                self.accept(before,new);failures=0
            else:failures+=1
            obs=new
            if failures>=2:return


def build(ctx):
    return Policy(ctx)
