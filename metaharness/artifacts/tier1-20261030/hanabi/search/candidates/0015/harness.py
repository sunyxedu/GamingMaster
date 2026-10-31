"""Threat-aware paired search over the public signaling protocol.

Every sampled world is private to the current decision; future actors receive
masked views. Public promises persist across those views exactly as in real
play. Exact joint inference uses small subset partition sums for speed.
"""
from __future__ import annotations
import json
import re
import random
import time
from collections import Counter

COLORS = 'RYGWB'
TYPES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = {x: (3 if x[1] == 1 else 1 if x[1] == 5 else 2) for x in TYPES}
FACES = list(TYPES)


def subsets(mask):
    sub = mask
    while True:
        yield sub
        if not sub:
            break
        sub = (sub - 1) & mask


def marginals(domains, counts):
    """Exact finite-population marginals by subset partition inversion.

    A block forces its slots to use the same physical card. Inclusion-exclusion
    over set partitions removes all such collisions; n <= 5 makes this small.
    This is algebraically the same posterior as the parent's matching DP.
    """
    n = len(domains)
    full = (1 << n) - 1
    if not n:
        return []
    allowed = {typ: sum(1 << i for i, dom in enumerate(domains) if typ in dom)
               for typ in TYPES if counts.get(typ, 0) > 0}
    moments = [0] * (full + 1)
    for typ, mask in allowed.items():
        sub = mask
        while sub:
            moments[sub] += counts[typ]
            sub = (sub - 1) & mask
    mu = (0, 1, -1, 2, -6, 24)
    z = [0] * (full + 1)
    z[0] = 1
    for mask in range(1, full + 1):
        root = mask & -mask
        rest = mask ^ root
        sub = rest
        while True:
            block = sub | root
            z[mask] += mu[block.bit_count()] * moments[block] * z[rest ^ sub]
            if not sub:
                break
            sub = (sub - 1) & rest
    total = z[full]
    result = [{} for _ in domains]
    if total <= 0:
        # Conservative fallback; never manufacture a zero-support identity.
        for i, dom in enumerate(domains):
            w = {x: counts.get(x, 0) for x in dom if counts.get(x, 0) > 0}
            den = sum(w.values())
            result[i] = {x: v / den for x, v in w.items()} if den else {}
        return result
    # Fix one physical card in slot i, then exclude that physical card from
    # every other slot. mu counts the ways the collision block can be formed.
    for typ, mask in allowed.items():
        cnt = counts[typ]
        bits = mask
        while bits:
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            optional = mask ^ bit
            sub = optional
            ways = 0
            while True:
                ways += mu[sub.bit_count() + 1] * z[full ^ bit ^ sub]
                if not sub:
                    break
                sub = (sub - 1) & optional
            if ways > 0:
                result[i][typ] = cnt * ways / total
    return result


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



class SimWorld:
    """Observation-derived hypothetical game. Never accesses game internals.

    Faces are sampled from ONE observer's information. Each simulated action
    is selected through view(), which masks that actor's actual/sampled hand.
    """
    def __init__(self, obs, own_faces, deck):
        self.h = dict(obs['fireworks'])
        self.tokens = obs['information_tokens']
        self.lives = obs['life_tokens']
        self.turn = obs['turn']
        self.seat = obs['observer']
        self.final = obs.get('final_turns_remaining')
        self.deck = list(deck)
        self.discards = [dict(x) for x in obs['discard_pile']]
        self.history = [dict(x) for x in obs.get('history', [])[-1:]]
        self.hands = [[dict(c, possible_colors=list(c['possible_colors']),
                            possible_ranks=list(c['possible_ranks'])) for c in hand]
                      for hand in obs['hands']]
        for card, face in zip(self.hands[self.seat], own_faces):
            card['color'], card['rank'] = face
        self.next_id = max([c['card_id'] for hand in self.hands for c in hand]
                           + [x['card_id'] for x in self.discards] + [-1]) + 1
        self.done = False

    def view(self):
        hands = []
        for s, hand in enumerate(self.hands):
            hands.append([dict(c, index=i, color=None if s == self.seat else c['color'],
                               rank=None if s == self.seat else c['rank'])
                          for i,c in enumerate(hand)])
        legal = [dict(type='PLAY',card_index=i) for i in range(len(hands[self.seat]))]
        if self.tokens < 8:
            legal += [dict(type='DISCARD',card_index=i) for i in range(len(hands[self.seat]))]
        if self.tokens:
            for s, hand in enumerate(self.hands):
                if s == self.seat:
                    continue
                legal += [dict(type='REVEAL_COLOR',target=s,color=c) for c in COLORS
                          if any(x['color']==c for x in hand)]
                legal += [dict(type='REVEAL_RANK',target=s,rank=r) for r in range(1,6)
                          if any(x['rank']==r for x in hand)]
        return dict(observer=self.seat,current_player=self.seat,num_players=len(hands),
                    turn=self.turn,fireworks=dict(self.h),information_tokens=self.tokens,
                    life_tokens=self.lives,deck_size=len(self.deck),final_turns_remaining=self.final,
                    hands=hands,discard_pile=[dict(x) for x in self.discards],
                    history=[dict(x) for x in self.history],
                    score=float(sum(self.h.values()) if self.lives > 0 else 0),
                    fireworks_sum=sum(self.h.values()),error=None,
                    legal_actions=[] if self.done else legal,done=self.done)

    def step(self, action):
        was_final = self.final is not None
        event = dict(action,turn=self.turn,actor=self.seat)
        kind = action['type']
        if kind.startswith('REVEAL_'):
            key = 'color' if kind == 'REVEAL_COLOR' else 'rank'
            value = action[key]
            mask = 'possible_colors' if key == 'color' else 'possible_ranks'
            ids = []
            for card in self.hands[action['target']]:
                touched = card[key] == value
                card[mask] = [v for v in card[mask] if (v == value) == touched]
                if touched:
                    card[key+'_clued'] = True
                    ids.append(card['card_id'])
            event['touched_card_ids'] = ids
            self.tokens -= 1
        else:
            card = self.hands[self.seat].pop(action['card_index'])
            c,r = card['color'],card['rank']
            event.update(card_id=card['card_id'],color=c,rank=r)
            success = kind == 'PLAY' and r == self.h[c]+1
            if success:
                self.h[c] += 1
                if r == 5:
                    self.tokens = min(8,self.tokens+1)
                event['outcome'] = 'played'
            else:
                self.discards.append(dict(card_id=card['card_id'],color=c,rank=r))
                if kind == 'PLAY':
                    self.lives -= 1
                    event['outcome'] = 'misplayed'
                else:
                    self.tokens = min(8,self.tokens+1)
                    event['outcome'] = 'discarded'
            self.done = self.lives <= 0 or sum(self.h.values()) == 25
            if not self.done and self.deck:
                c,r = self.deck.pop()
                self.hands[self.seat].append(dict(card_id=self.next_id,color=c,rank=r,
                    possible_colors=list(COLORS),possible_ranks=[1,2,3,4,5],
                    color_clued=False,rank_clued=False))
                event['drawn_card_id'] = self.next_id
                self.next_id += 1
                if not self.deck:
                    self.final = len(self.hands)
        if was_final:
            self.final -= 1
            if self.final <= 0:
                self.done = True
        self.turn += 1
        if not self.done:
            self.seat = (self.seat+1) % len(self.hands)
        self.history = [event]



class Policy:
    def __init__(self, ctx):
        self.ctx = ctx
        # This tracker contains ONLY deductions from public clue masks, actions,
        # stacks and the preagreed protocol. Never store visible faces here.
        self.promises = {}
        self.sessions = {}
        self.calls = 0
        self.last_call = -6
        self.simulating = False
        self._facts_cache = {}

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

    def protected(self, card):
        return card.get('color_clued') or card.get('rank_clued') or card['card_id'] in self.promises

    def facts(self, obs, seat, private=True):
        if getattr(self, '_cached_obs', None) is not obs:
            self._facts_cache = {}
            self._cached_obs = obs
        cache_key = (seat, private)
        if cache_key in self._facts_cache:
            return self._facts_cache[cache_key]
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
        self._facts_cache[cache_key] = facts
        return facts

    def clue_effect(self, obs, action, touched_ids=None):
        """Decode solely public information, exactly as the recipient will.

        Rank clues are literal and protective, never an implicit play command.
        Color clues promise the focus is playable NOW. Focus = oldest newly
        color-clued eligible card, else oldest eligible card. Eligible means
        the public literal/protocol domain permits the next rank of that color.
        If none is eligible, the clue is literal only.
        """
        hand = obs['hands'][action['target']]
        attr = 'color' if action['type'] == 'REVEAL_COLOR' else 'rank'
        value = action[attr]
        counts = remaining(obs)
        # Matching visible faces is allowed to the sender. The receiver uses
        # the engine's public touched IDs, yielding exactly the same set.
        touched = [i for i, card in enumerate(hand)
                   if (card[attr] == value if touched_ids is None
                       else card['card_id'] in touched_ids)]
        domains = []
        for i, card in enumerate(hand):
            dom = self.domain(card, counts)
            k = 0 if attr == 'color' else 1
            domains.append({x for x in dom if (x[k] == value) == (i in touched)})
        focus = None
        promise = None
        if attr == 'color':
            typ = (value, obs['fireworks'][value] + 1)
            eligible = [i for i in touched if typ in domains[i]]
            new = [i for i in eligible if not hand[i].get('color_clued')]
            if eligible:
                focus = (new or eligible)[0]
                promise = {typ}
                domains[focus] &= promise
        return touched, domains, focus, promise

    def discard_codebook(self, obs):
        """Pure public convention; invariant under erasure of ALL visible faces.

        Preserve the lowest-cost public alternative as an ordinary discard.
        Other alternatives are reserved as PLAY signals. If the
        partner already has a certain play, or cannot play again, all are neutral.
        Encoding depends on masks, public promises/counts, slot order and turn.
        """
        me = obs['observer']
        other = 1 - me
        own = self.facts(obs, me, private=False)
        partner = self.facts(obs, other, private=False)
        table = {i: None for i in range(len(own))}
        if obs.get('final_turns_remaining') == 1 or not own or not partner:
            return table
        if any(f['p_play'] > 1 - 1e-10 for f in partner):
            return table
        eligible = [f['index'] for f in partner if f['p_play'] > 1e-10]
        if not eligible:
            return table
        hand = obs['hands'][me]
        def public_cost(f):
            protected = self.protected(hand[f['index']]) and f['p_dead'] < 1-1e-10
            return f['loss'] + 0.45 * protected - 0.20 * (f['p_dead'] > 1-1e-10)
        ordered = sorted(own, key=lambda f: (public_cost(f), f['index']))
        reserved = ordered[1:]
        for j, f in enumerate(reserved):
            target_slot = eligible[(obs['turn'] // 2 + j) % len(eligible)]
            table[f['index']] = target_slot
        return table

    def public_view(self, obs):
        """A decoder snapshot containing no private observation or legal menu."""
        return {**{k: obs[k] for k in ('observer', 'turn', 'fireworks',
                                      'information_tokens', 'life_tokens',
                                      'deck_size', 'discard_pile', 'num_players')},
                'final_turns_remaining': obs.get('final_turns_remaining'),
                'hands': [[{**card, 'color': None, 'rank': None} for card in hand]
                          for hand in obs['hands']]}

    def accept_public_event(self, before, after):
        """Reconstruct communication from the actual accepted public event."""
        event = after.get('history', [])[-1:]
        if not event or event[0].get('turn') != before['turn']:
            return
        event = event[0]
        if event['type'].startswith('REVEAL_'):
            _, _, focus, promise = self.clue_effect(
                before, event, set(event.get('touched_card_ids', [])))
            if focus is not None:
                cid = before['hands'][event['target']][focus]['card_id']
                self.promises[cid] = promise
        elif event['type'] == 'DISCARD':
            target_slot = self.discard_codebook(before).get(event['card_index'])
            if target_slot is not None:
                target = 1 - before['observer']
                card = before['hands'][target][target_slot]
                promise = {x for x in self.domain(card, remaining(before))
                           if self.is_playable(x, before)}
                self.promises[card['card_id']] = promise
                if not self.simulating:
                    self.ctx.log('discard_play_received', sender=before['observer'],
                             turn=before['turn'], discarded_slot=event['card_index'],
                             target=target, target_id=card['card_id'],
                             possible=sorted(c+str(r) for c,r in promise))

    def candidates(self, obs):
        me = obs['observer']
        other = 1 - me
        counts = remaining(obs)
        cap = ceilings(obs, counts)
        own = self.facts(obs, me)
        partner = self.facts(obs, other, private=False)
        hand = obs['hands'][me]
        phand = obs['hands'][other]
        tokens = obs['information_tokens']
        deck = obs['deck_size']
        final = obs.get('final_turns_remaining')
        partner_safe = [f['index'] for f in partner if f['p_play'] > 1 - 1e-10]
        safe = [f['index'] for f in own if f['p_play'] > 1 - 1e-10]
        discard_table = self.discard_codebook(obs)
        chop_pool = [f for f in partner if not self.protected(phand[f['index']])]
        predicted = min(chop_pool or partner, key=lambda f: f['loss'] + 0.012 * f['index'])['index'] if partner else None
        candidates = []
        for action in obs['legal_actions']:
            kind = action['type']
            score = -99.0
            info = {}
            if kind == 'PLAY':
                i = action['card_index']
                f = own[i]
                pp = f['p_play']
                if pp > 1 - 1e-10:
                    unlock = sum(p * sum(card['color'] == x[0] and card['rank'] == x[1]+1
                                        for card in phand)
                                 for x,p in f['prob'].items())
                    five = sum(p for x,p in f['prob'].items() if x[1] == 5)
                    score = 2.6 + 0.35 * unlock + 0.25 * five
                    if tokens == 0 and partner_safe:
                        score -= 0.45
                    info = {'reason': 'certain play', 'unlocks_visible': round(unlock, 2)}
                elif (final is not None or deck <= 3) and obs['life_tokens'] > 1 and pp >= 0.35:
                    score = 1.2 * pp - (1-pp) * (0.25 + f['loss'])
                    if final == 1:
                        score = pp + 0.05
                    info = {'reason': 'late calculated risk', 'p_success': round(pp, 4)}
                elif final == 1 and pp > (obs['score'] / (obs['score'] + 1) if obs['score'] else 0.0):
                    score = (obs['score'] + 1) * pp - obs['score'] + 0.01
                    info = {'reason': 'final one-life expected score', 'p_success': round(pp, 4)}
                else:
                    continue
            elif kind == 'DISCARD':
                i = action['card_index']
                f = own[i]
                signal = discard_table[i]
                # A free play signal spends a real card.  Mean rollouts can
                # hide a low-probability unique card behind the signal's
                # immediate tempo gain, so suppress the channel when its
                # sender-side posterior already sees a material live-copy
                # threat.  The ordinary discard remains available.
                if signal is not None and f['p_critical'] >= 0.16 and f['p_dead'] < 0.75:
                    continue
                if signal is not None:
                    actual = (phand[signal]['color'], phand[signal]['rank'])
                    if not self.is_playable(actual, obs):
                        continue  # A legal discard can violate our public protocol.
                protection = 0.45 if self.protected(hand[i]) and f['p_dead'] < 1 - 1e-10 else 0
                score = 0.95 - 1.7 * f['loss'] - protection - 0.012 * i
                if tokens <= 1:
                    score += 0.45
                if f['p_dead'] > 1 - 1e-10:
                    score += 0.35
                if final is not None:
                    score = -0.15
                elif deck <= 2:
                    score -= 0.25
                info = {'reason': 'recover clue and draw', 'expected_loss': round(f['loss'],3),
                        'p_dead': round(f['p_dead'],3), 'p_critical': round(f['p_critical'],3)}
                if signal is not None:
                    # Use the same marginal actionable-play value as a clue.
                    # The base discard value already prices draw/token benefit
                    # and THIS seat's estimated cost of the removed card.
                    score += 2.5
                    info.update(reason='discard plus free play signal',
                                target_slot=signal, target_id=phand[signal]['card_id'])
            elif kind.startswith('REVEAL_'):
                touched, after, focus, promise = self.clue_effect(obs, action)
                if focus is not None:
                    actual = (phand[focus]['color'], phand[focus]['rank'])
                    if actual not in promise:
                        continue  # Truthful literal clues may still violate the protocol.
                enabled = [i for i, dom in enumerate(after) if dom and all(self.is_playable(x, obs) for x in dom)]
                newly = [i for i in enabled if i not in partner_safe]
                unique = len({(phand[i]['color'], phand[i]['rank']) for i in newly})
                dead_new = sum(bool(dom) and all(self.is_dead(x,obs,cap) for x in dom)
                               and partner[i]['p_dead'] < 1-1e-10 for i,dom in enumerate(after))
                protect = 0.0
                saved = []
                for i in touched:
                    card = phand[i]
                    typ = (card['color'],card['rank'])
                    if counts[typ] == 1 and not self.is_dead(typ,obs,cap) and not self.protected(card):
                        saved.append(i)
                        if not partner_safe:
                            protect = max(protect, (2.0 if i == predicted else 0.20) *
                                          min(2.0, (cap[typ[0]] - typ[1] + 1) / 2))
                score = -0.6 + 2.5 * min(unique, 1) + 0.65 * max(0, unique-1) + 0.15 * dead_new + protect
                if partner_safe:
                    score -= 0.8
                if tokens == 1:
                    score -= 0.45
                # Literal preparation helps when color focus is obstructed.
                information = sum(len(partner[i]['domain']) - len(dom) for i,dom in enumerate(after))
                score += min(0.25, information * 0.008)
                if tokens >= 6:
                    score += 0.2
                if final is not None:
                    score = 0.8 if final >= 2 and newly and not partner_safe else -0.3
                info = {'reason': 'play signal' if focus is not None else 'literal/save clue',
                        'focus': focus, 'new_safe_slots': newly, 'save_unique_slots': saved,
                        'predicted_partner_discard': predicted, 'partner_already_safe': partner_safe}
            else:
                continue
            candidates.append({'action': action, 'value': round(score, 5), **info})
        candidates.sort(key=lambda x: x['value'], reverse=True)
        return candidates, own, safe

    def urgent_save(self, obs):
        """Return a literal save clue when the public chop would erase a live chain.

        The ordinary scalar utility can undervalue a save when its rollout
        estimate has high variance.  This gate uses only the sender's legal
        view: predict the recipient's public chop, require that no safe play
        exists, and protect a unique, still-reachable card before ranking any
        speculative action.  Rank clues are deliberately preferred because
        they carry no play promise and remain a sound save convention.
        """
        if obs.get('information_tokens', 0) <= 0 or obs.get('final_turns_remaining') is not None:
            return None
        me = obs['observer']; other = 1 - me
        phand = obs['hands'][other]
        if not phand:
            return None
        partner = self.facts(obs, other, private=False)
        if any(f['p_play'] > 1 - 1e-10 for f in partner):
            return None
        chop_pool = [f for f in partner if not self.protected(phand[f['index']])]
        if not chop_pool:
            return None
        predicted = min(chop_pool, key=lambda f: f['loss'] + 0.012 * f['index'])
        card = phand[predicted['index']]
        typ = (card.get('color'), card.get('rank'))
        if typ[0] not in COLORS or typ[1] not in range(1, 6):
            return None
        counts = remaining(obs); cap = ceilings(obs, counts)
        if counts.get(typ, 0) != 1 or self.is_dead(typ, obs, cap):
            return None
        # A clue touching this exact card is always truthful.  Use a rank
        # clue first; fall back to its color if the rank command is unavailable.
        choices = []
        for action in obs.get('legal_actions', []):
            if action.get('type') == 'REVEAL_RANK' and action.get('target') == other \
                    and action.get('rank') == typ[1]:
                choices.append(action)
        for action in obs.get('legal_actions', []):
            if action.get('type') == 'REVEAL_COLOR' and action.get('target') == other \
                    and action.get('color') == typ[0]:
                choices.append(action)
        return choices[0] if choices else None

    def sample_world(self, obs, rng):
        """Exact sequential conditional draw from literal plus public promises."""
        me = obs['observer']
        counts = remaining(obs)
        for s, hand in enumerate(obs['hands']):
            if s != me:
                for card in hand:
                    counts[card['color'], card['rank']] -= 1
        domains = [self.domain(c, counts) for c in obs['hands'][me]]
        faces = []
        for i in range(len(domains)):
            prob = marginals(domains[i:], counts)[0]
            if not prob:
                return None
            # Sorting makes the sampling stream reproducible across hash seeds.
            choices = sorted(prob)
            pick = rng.random() * sum(prob.values())
            chosen = choices[-1]
            for face in choices:
                pick -= prob[face]
                if pick <= 0:
                    chosen = face
                    break
            faces.append(chosen)
            counts[chosen] -= 1
        if any(v < 0 for v in counts.values()):
            return None
        deck = [x for x in TYPES for _ in range(counts[x])]
        if len(deck) != obs['deck_size']:
            return None
        rng.shuffle(deck)
        return faces, deck

    def finish_world(self, obs, sampled, action, stop):
        world = SimWorld(obs, *sampled)
        sim = Policy(self.ctx)
        sim.simulating = True
        # Every entry is reconstructible from public actions. No private state
        # or sessions are copied to a simulated actor.
        sim.promises = dict(self.promises)
        for step in range(180):
            if stop():
                return None
            view = world.view()
            if step:
                options = sim.candidates(view)[0]
                if not options:
                    return None
                action = options[0]['action']
            before = sim.public_view(view)
            world.step(action)
            after = world.view()
            sim.accept_public_event(before, after)
            ids = {c['card_id'] for h in after['hands'] for c in h}
            sim.promises = {cid: dom for cid, dom in sim.promises.items() if cid in ids}
            if world.done:
                return float(sum(world.h.values()) if world.lives > 0 else 0)
        return None

    def rollout_search(self, obs, entries):
        """Evaluate the unchanged policy on paired, information-safe worlds."""
        if len(entries) < 2 or obs.get('num_players') != 2:
            return entries
        budget = self.ctx.budget.remaining()
        if budget.get('wall_s', 0) < 45:
            return entries
        # A dominant local safe play rarely benefits from expensive arbitration.
        if (entries[0]['reason'] == 'certain play'
                and entries[0]['value'] - entries[1]['value'] > 0.8):
            return entries
        # Keep two options per action family; cover opportunity costs even when
        # a play or save is outside the parent's top six scalar utilities.
        options = [dict(entries[0])]
        chosen_actions = [entries[0]['action']]
        for family in ('PLAY', 'DISCARD', 'REVEAL_COLOR', 'REVEAL_RANK'):
            picked = 0
            for e in entries:
                if e['action']['type'] != family:
                    continue
                if e['action'] not in chosen_actions:
                    options.append(dict(e)); chosen_actions.append(e['action'])
                picked += 1
                if picked >= 2:
                    break
        if len(options) < 2:
            return entries
        seconds = 12.0 if obs['deck_size'] <= 12 else 8.0
        deadline = time.monotonic() + min(seconds, budget['wall_s'] - 30)
        stop = lambda: self.ctx.done or time.monotonic() >= deadline
        rng = random.Random(9173 + obs['turn'] * 1009 + obs['observer'])
        totals = [0.0] * len(options)
        delta_totals = [0.0] * len(options)
        delta_sq = [0.0] * len(options)
        n = 0
        for _ in range(64):
            if stop():
                break
            sampled = self.sample_world(obs, rng)
            if sampled is None:
                break
            values = []
            for option in options:
                val = self.finish_world(obs, sampled, option['action'], stop)
                if val is None:
                    break
                values.append(val)
            if len(values) != len(options):
                break  # Do not compare partial or horizon-truncated games.
            n += 1
            for i, val in enumerate(values):
                delta = val - values[0]
                totals[i] += val
                delta_totals[i] += delta
                delta_sq[i] += delta * delta
        if n < 12:
            self.ctx.log('search_insufficient', turn=obs['turn'], samples=n,
                         options=len(options))
            return entries
        for i, e in enumerate(options):
            gain = delta_totals[i] / n
            se = (max(0.0, delta_sq[i] / n - gain * gain) / (n - 1)) ** .5
            e['rollout_score'] = round(totals[i] / n, 4)
            e['rollout_n'] = n
            e['paired_gain'] = round(gain, 4)
            e['paired_se'] = round(se, 4)
            e['evidence'] = gain - 1.5 * se - .15 if i else 0.0
            e['parent_choice'] = i == 0
        options.sort(key=lambda e: (e['evidence'], e['value']), reverse=True)
        self.ctx.log('paired_rollout', turn=obs['turn'], samples=n, options=options)
        # Keep baseline and evidence-qualified alternatives in the advisory menu.
        # Every option is already legal and checked by the unchanged protocol.
        qualified = [e for e in options if e['evidence'] >= 0]
        return qualified

    def ask(self, obs, candidates, facts):
        ctx = self.ctx
        remaining_budget = ctx.budget.remaining()
        if (self.calls >= 3 or remaining_budget['system2_calls'] <= 2 or
                remaining_budget['wall_s'] < 900 or obs['turn'] - self.last_call < 5):
            return None
        if len(candidates) < 2 or 'rollout_score' not in candidates[0]:
            return None
        options = candidates[:6]
        # All model context is current-seat-only. Public tracker carries no private faces.
        seat = obs['observer']
        state = {'seat': seat, 'turn': obs['turn'], 'fireworks': obs['fireworks'],
                 'clues': obs['information_tokens'], 'lives': obs['life_tokens'],
                 'deck': obs['deck_size'], 'final_actions': obs.get('final_turns_remaining'),
                 'own': [{'slot': i, 'id': card['card_id'],
                          'p_play': round(facts[i]['p_play'],3),
                          'p_dead': round(facts[i]['p_dead'],3),
                          'discard_loss': round(facts[i]['loss'],3),
                          'possible': [c+str(r) for c,r in sorted(facts[i]['domain'])]}
                         for i,card in enumerate(obs['hands'][seat])],
                 'partner_visible': obs['hands'][1-seat],
                 'public_remaining': {c+str(r): n for (c,r),n in remaining(obs).items()},
                 'options': [{'id': i, **candidate} for i,candidate in enumerate(options)]}
        prompt = (
            'Choose a strategic action for two-player cooperative Hanabi. Maximize final score 0-25; '
            'third misplay gives ZERO. Code handles inference and ordinary turns. You are ONLY seat '
            + str(seat) + '. Never invent your own card faces. The supplied candidates are legal and '
            'protocol-validated; choose only a listed option ID. RANK clues give literal information '
            'and protect touched cards. COLOR clues promise the indicated focus is immediately playable; '
            'all other touches are literal only. Some discard indices ALSO promise a safe partner play, '
            'according to a fixed public table; these candidates say discard plus free play signal. '
            'Their target slot has been verified safe, and the discarded card loss is estimated. '
            'The recipient executes certain plays automatically. '
            'Do not invent finesse or unstated conventions. Preserve critical low ranks, prefer actionable '
            'clues, avoid redundant clues and over-saving high cards; draw through the deck. '
            'rollout_score estimates terminal score under the same protocol, with masked future seats. '
            'Paired gains compare the parent option in identical sampled worlds; the evidence margin '
            'accounts for sampling error, not policy-selection bias in beliefs. Only alternatives passing '
            'that margin plus the parent are listed. Prefer the largest evidence unless a concrete '
            'strategic defect explains its value. '
            'Reply ONLY with JSON {"choice": integer, "reason": "at most 15 words"}.\n' +
            json.dumps(state, separators=(',', ':')))
        self.calls += 1
        self.last_call = obs['turn']
        try:
            # The outer wait includes dispatch/queue time, which the backend's
            # own timeout may exclude. Only ctx-managed model work is spawned;
            # the evaluator cancels it when autonomous play ends the episode.
            session = self.sessions.get(seat)
            future = ctx.spawn(lambda: ctx.think(
                prompt, session_id=session,
                timeout_s=min(90, remaining_budget['wall_s'] - 600)))
            result = future.result(timeout=120)
            self.sessions[seat] = result.session_id
            match = re.search(r'\{[^{}]*"choice"\s*:\s*\d+[^{}]*\}', result.text)
            if match:
                choice = json.loads(match.group())['choice']
            elif re.fullmatch(r'\s*\d+\s*', result.text):
                choice = int(result.text)
            else:
                return None
            if isinstance(choice, int) and 0 <= choice < len(options):
                return options[choice]
        except Exception as exc:
            # Avoid starting more inference with a possibly outstanding request.
            self.calls = 3
            ctx.log('planner_fallback', error=type(exc).__name__, seat=seat)
        return None

    def run_episode(self, game):
        ctx = self.ctx
        # Be correct even if build's object is reused across reset by a caller.
        self.promises = {}
        self.sessions = {}
        self.calls = 0
        self.last_call = -6
        self._cached_obs = None
        self._facts_cache = {}
        obs = ctx.observe()
        failures = 0
        for _ in range(220):
            if ctx.done or obs.get('done') or not obs.get('legal_actions'):
                break
            candidates, facts, safe = self.candidates(obs)
            rescue = self.urgent_save(obs)
            if rescue is not None:
                # This is a structural safety gate, rather than a scalar
                # score tweak: do not spend rollout time on a move that would
                # publicly chop the only reachable copy of a card.
                selected = {'action': rescue, 'reason': 'urgent critical save'}
                self.ctx.log('urgent_save', action=rescue, seat=obs['observer'],
                             turn=obs['turn'])
                candidates = []
            elif candidates:
                candidates = self.rollout_search(obs, candidates)
            if rescue is None and not candidates:
                # At full clues a rank clue is always available and literal-only.
                legal = obs['legal_actions']
                action = next((a for a in legal if a['type']=='REVEAL_RANK'), legal[0])
                selected = {'action': action, 'reason': 'legal fallback'}
            elif rescue is None:
                selected = candidates[0]
                if not safe and failures == 0:
                    selected = self.ask(obs, candidates, facts) or selected
            if ctx.done:
                break
            action = selected['action']
            before = self.public_view(obs)
            self._cached_obs = None
            self._facts_cache = {}
            try:
                new_obs = ctx.act(action)
            except Exception as exc:
                if ctx.done:
                    break
                failures += 1
                ctx.log('action_failure', error=type(exc).__name__)
                if failures >= 3:
                    break
                obs = ctx.observe()
                continue
            if not isinstance(new_obs, dict):
                new_obs = ctx.observe()
            if new_obs.get('turn', -1) > obs.get('turn', -1):
                self.accept_public_event(before, new_obs)
                event = new_obs.get('history', [])[-1:]
                if event and event[0].get('type') == 'DISCARD':
                    face = (event[0]['color'], event[0]['rank'])
                    counts = remaining(before)
                    cap = ceilings(before, counts)
                    if (counts[face] == 1 and before['fireworks'][face[0]] < face[1]
                            <= cap[face[0]]):
                        ctx.log('lost_live_last_copy', turn=before['turn'],
                                face=face[0] + str(face[1]),
                                ceiling_points=cap[face[0]] - face[1] + 1)
                active_ids = {card['card_id'] for hand in new_obs['hands'] for card in hand}
                self.promises = {cid: dom for cid,dom in self.promises.items() if cid in active_ids}
                failures = 0
            else:
                failures += 1
            ctx.log('decision', seat=obs['observer'], turn=obs['turn'], selected=selected,
                    planner_calls=self.calls)
            obs = new_obs
            if failures >= 3:
                break


def build(ctx):
    return Policy(ctx)
