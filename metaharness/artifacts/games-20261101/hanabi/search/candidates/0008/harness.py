"""Action-choice communication: a discard can restore a clue AND promise a play.

Discard slots form a small, preagreed codebook with one neutral alternative.
The remaining slots signal a partner play. Private discard costs decide whether
transmitting is worthwhile. Senders verify promises from their permitted view.
Only public domains, never a sender's private faces, survive an action.
"""
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
        self.last_call = -6

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

    def ask(self, obs, candidates, facts):
        ctx = self.ctx
        remaining_budget = ctx.budget.remaining()
        if (self.calls >= 3 or remaining_budget['system2_calls'] <= 2 or
                remaining_budget['wall_s'] < 900 or obs['turn'] - self.last_call < 5):
            return None
        if len(candidates) < 2 or candidates[0]['value'] - candidates[1]['value'] > 1.2:
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
            'The values are a heuristic, not a search result; override for a concrete strategic reason. '
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
        obs = ctx.observe()
        failures = 0
        for _ in range(220):
            if ctx.done or obs.get('done') or not obs.get('legal_actions'):
                break
            candidates, facts, safe = self.candidates(obs)
            if not candidates:
                # At full clues a rank clue is always available and literal-only.
                legal = obs['legal_actions']
                action = next((a for a in legal if a['type']=='REVEAL_RANK'), legal[0])
                selected = {'action': action, 'reason': 'legal fallback'}
            else:
                selected = candidates[0]
                if not safe and failures == 0:
                    selected = self.ask(obs, candidates, facts) or selected
            if ctx.done:
                break
            action = selected['action']
            before = self.public_view(obs)
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
