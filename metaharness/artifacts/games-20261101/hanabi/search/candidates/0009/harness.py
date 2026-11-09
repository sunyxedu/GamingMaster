"""Discard-to-play communication: a public action codebook with seat-local beliefs."""
from __future__ import annotations
import json
import re
from collections import Counter

COLORS = 'RYGWB'
TYPES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = {x: (3 if x[1] == 1 else 1 if x[1] == 5 else 2) for x in TYPES}


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


class Policy:
    def __init__(self, ctx):
        self.ctx = ctx
        # This tracker contains ONLY deductions from public clue masks, actions,
        # stacks and the preagreed protocol. Never store visible faces here.
        self.promises = {}
        self.sessions = {}
        self.calls = 0
        self.last_call = -12

    def domain(self, card, counts):
        dom = {x for x in literal(card) if counts.get(x, 0) > 0}
        promise = self.promises.get(card['card_id'])
        if promise is not None:
            narrowed = dom & promise
            if narrowed:
                dom = narrowed
        return dom

    def belief(self, obs, seat, private=True):
        counts = remaining(obs)
        if private:
            # Only subtract cards visible to THIS observer.
            for s, hand in enumerate(obs['hands']):
                if s != seat:
                    for card in hand:
                        if card['color'] is not None:
                            counts[card['color'], card['rank']] -= 1
        domains = [self.domain(card, counts) for card in obs['hands'][seat]]
        return marginals(domains, counts), domains

    def is_playable(self, typ, obs):
        return typ[1] == obs['fireworks'][typ[0]] + 1

    def is_dead(self, typ, obs, cap):
        return typ[1] <= obs['fireworks'][typ[0]] or typ[1] > cap[typ[0]]

    def facts(self, obs, seat, private=True):
        counts = remaining(obs)
        cap = ceilings(obs, counts)
        belief, domains = self.belief(obs, seat, private)
        visible = Counter()
        if private:
            for s, hand in enumerate(obs['hands']):
                if s != seat:
                    for card in hand:
                        visible[card['color'], card['rank']] += 1
        facts = []
        for i, prob in enumerate(belief):
            pp = pd = pc = loss = 0.0
            for typ, p in prob.items():
                dead = self.is_dead(typ, obs, cap)
                pp += p * self.is_playable(typ, obs)
                pd += p * dead
                critical = counts[typ] == 1 and not dead
                pc += p * critical
                if not dead:
                    distance = typ[1] - obs['fireworks'][typ[0]]
                    if critical:
                        cost = (cap[typ[0]] - typ[1] + 1) * (0.9 ** (distance - 1))
                    else:
                        cost = 0.42 / counts[typ] + 0.22 / max(1, distance)
                        if visible[typ]:
                            cost *= 0.25
                    loss += p * cost
            facts.append({'index': i, 'p_play': pp, 'p_dead': pd, 'p_critical': pc,
                          'loss': loss, 'domain': domains[i], 'prob': prob})
        return facts

    def public_chop(self, obs, seat, facts=None):
        """No private face or private card-count subtraction affects the schedule."""
        facts = facts if facts is not None else self.facts(obs, seat, private=False)
        if not facts:
            return None
        dead = [f for f in facts if f['p_dead'] > 1 - 1e-9]
        if dead:
            return min(dead, key=lambda f: f['index'])['index']
        hand = obs['hands'][seat]
        pool = [f for f in facts if hand[f['index']]['card_id'] not in self.promises
                and f['p_critical'] < 1-1e-9 and f['p_play'] < 1-1e-9]
        # Clue flags alone do not protect a card forever. A save promise does.
        return min(pool or facts, key=lambda f: (f['loss'] + 0.015*f['index'], f['index']))['index']

    def clue_effect(self, obs, action, touched_ids=None):
        """Public decoder. Optional IDs let the receiver avoid sender-visible faces.

        Color: same oldest eligible play focus as parent 0001.
        Rank 2..5 touching public chop, below 8 tokens: critical-save promise.
        Other rank clues below 8 tokens are literal only.
        At 8 tokens rank clues encode a direct play; colors are literal escape.
        """
        target = action['target']
        hand = obs['hands'][target]
        attr = 'color' if action['type'] == 'REVEAL_COLOR' else 'rank'
        value = action[attr]
        counts = remaining(obs)
        cap = ceilings(obs, counts)
        if touched_ids is None:
            touched_ids = {c['card_id'] for c in hand if c[attr] == value}
        touched = [i for i,c in enumerate(hand) if c['card_id'] in touched_ids]
        domains = []
        axis = 0 if attr == 'color' else 1
        for i, card in enumerate(hand):
            domains.append({x for x in self.domain(card, counts)
                            if (x[axis] == value) == (i in touched)})
        focus = None
        promise = None
        mode = 'literal'
        if obs['information_tokens'] == 8:
            # Discard signaling is unavailable at the token cap. Rank symbols
            # become direct play packets; color clues are literal escape symbols.
            # Thus at least one truthful non-signaling clue always exists.
            if attr == 'rank':
                eligible = [i for i, card in enumerate(hand)
                            if any(self.is_playable(x, obs)
                                   for x in self.domain(card, counts))]
                if eligible:
                    focus = eligible[(value - 1 + obs['turn'] // 2) % len(eligible)]
                    promise = {x for x in domains[focus] if self.is_playable(x, obs)}
                    mode = 'overflow play'
        elif attr == 'color':
            typ = (value, obs['fireworks'][value] + 1)
            eligible = [i for i in touched if typ in domains[i]]
            fresh = [i for i in eligible if not hand[i].get('color_clued')]
            if eligible:
                focus = (fresh or eligible)[0]
                promise = {typ}
                mode = 'play'
        elif value >= 2 and obs['information_tokens'] < 8:
            chop = self.public_chop(obs, target)
            if chop in touched:
                focus = chop
                promise = {x for x in domains[chop] if counts[x] == 1
                           and not self.is_dead(x, obs, cap)}
                mode = 'save'
        if focus is not None:
            domains[focus] &= promise
        return touched, domains, focus, promise, mode

    def discard_codebook(self, obs, actor, own_public=None, partner_public=None):
        """Map discard slots to recipient slots using ONLY public information.

        The public chop is always neutral. Every other mapped slot certifies a
        playable recipient card, even when the discard face itself is unknown.
        Publicly impossible/already-certain plays consume no code symbols.
        """
        other = 1 - actor
        own = own_public if own_public is not None else self.facts(obs, actor, private=False)
        partner = partner_public if partner_public is not None else self.facts(obs, other, private=False)
        neutral = self.public_chop(obs, actor, own)
        size = len(partner)
        if not size or obs.get('final_turns_remaining') == 1:
            return neutral, {}
        offset = (obs['turn'] // 2) % size
        targets = [f for f in partner if 1e-10 < f['p_play'] < 1 - 1e-9]
        targets.sort(key=lambda f: (-round(f['p_play'], 8), (f['index'] - offset) % size))
        if not targets:
            return neutral, {}
        sources = [f for f in own if f['index'] != neutral]
        sources.sort(key=lambda f: (f['loss'], f['index']))
        return neutral, {f['index']: targets[j % len(targets)]['index']
                         for j, f in enumerate(sources)}

    def discard_effect(self, obs, action):
        """Reconstruct a public promise without inspecting either hand's faces."""
        me = obs['observer']
        _, mapping = self.discard_codebook(obs, me)
        focus = mapping.get(action['card_index'])
        if focus is None:
            return None
        card = obs['hands'][1-me][focus]
        counts = remaining(obs)
        promise = {x for x in self.domain(card, counts) if self.is_playable(x, obs)}
        return card['card_id'], promise

    def candidates(self, obs):
        me = obs['observer']
        other = 1 - me
        counts = remaining(obs)
        cap = ceilings(obs, counts)
        own = self.facts(obs, me)
        public_own = self.facts(obs, me, private=False)
        partner = self.facts(obs, other, private=False)
        hand, phand = obs['hands'][me], obs['hands'][other]
        tokens, deck = obs['information_tokens'], obs['deck_size']
        final = obs.get('final_turns_remaining')
        safe = [f['index'] for f in own if f['p_play'] > 1-1e-9]
        psafe = [f['index'] for f in partner if f['p_play'] > 1-1e-9]
        chop = self.public_chop(obs, me, public_own)
        pchop = self.public_chop(obs, other, partner)
        threatened = None
        if pchop is not None:
            c = phand[pchop]
            x = c['color'], c['rank']
            if counts[x] == 1 and not self.is_dead(x, obs, cap) and not psafe:
                threatened = x
        neutral, discard_signals = self.discard_codebook(obs, me, public_own, partner)
        entries = []
        for action in obs['legal_actions']:
            kind = action['type']
            score = -100
            reason = ''
            details = {}
            if kind == 'PLAY':
                i = action['card_index']; f = own[i]; pp = f['p_play']
                if pp > 1-1e-9:
                    unlock = sum(p * any(c['color'] == x[0] and c['rank'] == x[1]+1
                                       for c in phand) for x,p in f['prob'].items())
                    five = sum(p for x,p in f['prob'].items() if x[1] == 5)
                    score = 5 + .5*unlock + .35*five
                    # Check whether this play makes the partner publicly safe.
                    prevents = 0.0
                    for x,p in f['prob'].items():
                        h = dict(obs['fireworks']); h[x[0]] += 1
                        if any(g['domain'] and all(y[1] == h[y[0]]+1 for y in g['domain'])
                               for g in partner):
                            prevents += p
                    if threatened and prevents < 1-1e-9 and tokens:
                        score -= 3.5
                    reason = 'certain play'
                elif final is not None:
                    gain = pp if obs['life_tokens'] > 1 else pp-(1-pp)*sum(obs['fireworks'].values())
                    if gain <= 0:
                        continue
                    score = gain + (0.2 if final == 1 else -0.25)
                    reason = 'final expected score risk'
                else:
                    continue
                details = {'p_success': round(pp, 5)}
            elif kind == 'DISCARD':
                i = action['card_index']; f = own[i]
                focus = discard_signals.get(i)
                if focus is not None:
                    actual = phand[focus]['color'], phand[focus]['rank']
                    if not self.is_playable(actual, obs):
                        continue  # Non-neutral discards MUST tell the truth.
                    # A token-positive play signal competes with an ordinary clue.
                    # It still pays the expected loss of the sender's hidden card.
                    score = 4.5 - 2.7*f['loss'] + .35*f['p_dead']
                    if tokens <= 2:
                        score += .55
                    if psafe:
                        score -= 1.8
                    if threatened:
                        score += 2.6  # An immediate play prevents the chop.
                    if final is not None:
                        score = .9 - .1*f['loss'] if not psafe else -.25
                    reason = 'discard play signal'
                    details = {'expected_loss': round(f['loss'],4), 'signal_slot':focus,
                               'signal_card_id':phand[focus]['card_id'], 'neutral_slot':neutral}
                else:
                    # At least the public chop always remains an escape symbol.
                    # Publicly unassigned symbols may carry no instruction.
                    if i != chop and f['p_dead'] < 1-1e-9:
                        continue
                    score = 1.05 - 1.7*f['loss'] + .45*f['p_dead']
                    if tokens <= 1:
                        score += .9
                    if final is not None:
                        score = -.25
                    elif deck <= 2:
                        score -= .4
                    if threatened and tokens:
                        score -= 3.5
                    reason = 'neutral discard'
                    details = {'expected_loss':round(f['loss'],4), 'neutral_slot':neutral}
            else:
                touched, domains, focus, promise, mode = self.clue_effect(obs, action)
                if focus is not None:
                    actual = phand[focus]['color'], phand[focus]['rank']
                    if not promise or actual not in promise:
                        continue
                enabled = [i for i,d in enumerate(domains) if d and all(self.is_playable(x,obs) for x in d)]
                newly = [i for i in enabled if i not in psafe]
                distinct = len({(phand[i]['color'],phand[i]['rank']) for i in newly})
                # A save can itself identify a next card: count it as productive.
                score = -.6 + 3.4*min(1,distinct) + .85*max(0,distinct-1)
                if psafe:
                    score -= 1.0
                if tokens == 1:
                    score -= .5
                if tokens >= 6:
                    score += .3
                information = sum(len(partner[i]['domain'])-len(d) for i,d in enumerate(domains))
                score += min(.3, .008*information)
                saves = mode == 'save' and focus == pchop
                if saves:
                    x = phand[focus]['color'],phand[focus]['rank']
                    # Save before the next scheduled discard, including when our
                    # own scoring play would otherwise pass the endangered hand.
                    score += (5.8 + .35*(cap[x[0]]-x[1])) if threatened else .3
                if threatened and newly:
                    score += 3.5  # An executable play also avoids that discard.
                if final is not None:
                    score = (.85 + .1*distinct) if final >= 2 and newly and not psafe else -.6
                reason = mode + ' clue'
                details = {'focus': focus, 'new_safe': newly, 'saves_chop': saves,
                           'partner_chop': pchop}
            entries.append({'action':action,'value':round(score,5),'reason':reason,**details})
        entries.sort(key=lambda e:e['value'], reverse=True)
        return entries, own, safe

    def ask(self, obs, entries, own):
        budget = self.ctx.budget.remaining()
        # Advice only ranks already validated actions. The discard agreement is
        # enforced in code and cannot be rewritten by either private session.
        options = [e for e in entries[:5] if e['value'] >= entries[0]['value']-.65]
        if (self.calls >= 2 or len(options)<2 or obs['turn']-self.last_call<12
                or budget.get('system2_calls',0)<=2 or budget.get('wall_s',0)<500
                or obs.get('final_turns_remaining') is not None
                or entries[0]['reason'] == 'certain play'):
            return None
        seat=obs['observer']
        view = {'seat':seat,'turn':obs['turn'],'stacks':obs['fireworks'],
                'tokens':obs['information_tokens'],'deck':obs['deck_size'],
                'own':[{'slot':f['index'],'play':round(f['p_play'],3),
                        'loss':round(f['loss'],3),'possible':[c+str(r) for c,r in sorted(f['domain'])]}
                       for f in own],
                'partner_visible':obs['hands'][1-seat],
                'options':[{'choice':i,**e} for i,e in enumerate(options)]}
        prompt = ('Choose one legal Hanabi option to maximize terminal score. Third misplay scores zero. '
                  'Your session belongs only to seat '+str(seat)+'. Code uses a PUBLIC codebook: '
                  'neutral discard carries no information; discard play signal certifies the indicated '
                  'partner slot immediately playable and restores a token. It pays the expected loss '
                  'of the sender hidden card, so compare information/token benefit against that loss. '
                  'At 8 clues, rank clues signal a safe play and color clues are literal escape symbols. '
                  'Below 8 clues, color clues promise oldest eligible focus playable now, and rank 2..5 touching '
                  'public chop saves a live last copy; other rank clues are literal. All promises are verified. Prefer '
                  'productive signals, protect low prerequisites, and finish held chains before deck exhaustion. '
                  'Reply only {"choice": INTEGER}.\n'+json.dumps(view,separators=(',',':')))
        self.calls += 1; self.last_call=obs['turn']
        try:
            session=self.sessions.get(seat)
            future=self.ctx.spawn(lambda: self.ctx.think(prompt,session_id=session,timeout_s=75))
            result=future.result(timeout=80)
            self.sessions[seat]=result.session_id
            match=re.search(r'"choice"\s*:\s*(\d+)',result.text)
            if match and 0 <= int(match.group(1)) < len(options):
                return options[int(match.group(1))]
        except Exception as exc:
            self.calls=2
            self.ctx.log('adviser_fallback',error=type(exc).__name__,seat=seat)
        return None

    def run_episode(self, game):
        self.promises={};self.sessions={};self.calls=0;self.last_call=-12
        ctx=self.ctx;obs=ctx.observe();failures=0
        for _ in range(240):
            if ctx.done or obs.get('done') or not obs.get('legal_actions'):
                return
            entries, own, safe=self.candidates(obs)
            if not entries:
                return
            selected=self.ask(obs,entries,own) or entries[0]
            if ctx.done:
                return
            action=selected['action']
            ctx.log('decision',seat=obs['observer'],turn=obs['turn'],selected=selected,
                    promises={str(cid):[c+str(r) for c,r in sorted(d)] for cid,d in self.promises.items()})
            try:
                after=ctx.act(action)
            except Exception as exc:
                if ctx.done:
                    return
                failures+=1
                if failures>=2:
                    return
                obs=ctx.observe();continue
            if not isinstance(after,dict):
                after=ctx.observe()
            if after.get('turn',-1)>obs['turn']:
                # Decode from PUBLIC touched IDs. Sender-visible faces are never
                # copied into shared memory, even though sender checked truth.
                if action['type'].startswith('REVEAL_'):
                    event=after['history'][-1]
                    _,_,focus,promise,mode=self.clue_effect(obs,action,set(event['touched_card_ids']))
                    if focus is not None and promise:
                        cid=obs['hands'][action['target']][focus]['card_id']
                        old=self.promises.get(cid)
                        self.promises[cid]=promise if old is None else old & promise
                if action['type'] == 'DISCARD':
                    decoded = self.discard_effect(obs, action)
                    if decoded is not None:
                        cid, promise = decoded
                        old = self.promises.get(cid)
                        self.promises[cid] = promise if old is None else old & promise
                        ctx.log('discard_signal_decoded',turn=obs['turn'],card_id=cid,
                                domain=[c+str(r) for c,r in sorted(promise)])
                event=after['history'][-1]
                if event['type'] == 'DISCARD':
                    x=event['color'],event['rank']
                    before_counts=remaining(obs)
                    before_cap=ceilings(obs,before_counts)
                    if before_counts[x] == 1 and not self.is_dead(x,obs,before_cap):
                        ctx.log('lost_live_last_copy',turn=event['turn'],face=x[0]+str(x[1]),
                                ceiling_points=before_cap[x[0]]-x[1]+1,
                                tokens_before=obs['information_tokens'])
                active={c['card_id'] for h in after['hands'] for c in h}
                self.promises={cid:d for cid,d in self.promises.items() if cid in active}
                failures=0
            else:
                failures+=1
                if failures>=2:
                    return
            obs=after


def build(ctx):
    return Policy(ctx)
