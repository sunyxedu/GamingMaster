"""Public direct-play promises, exact finite-population beliefs, sparse Astra advice.

Private observations are never stored across seats.  The only cross-seat state
is decoded from public events with a fixed protocol; model sessions are separate.
"""
from __future__ import annotations
import json
import re
from collections import Counter

COLORS = 'RYGWB'
FACES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = {x: (3 if x[1] == 1 else 1 if x[1] == 5 else 2) for x in FACES}


def public_counts(obs):
    counts = COPIES.copy()
    for c, height in obs['fireworks'].items():
        for r in range(1, height + 1):
            counts[c, r] -= 1
    for card in obs.get('discard_pile', []):
        counts[card['color'], card['rank']] -= 1
    return counts


def ceilings(obs, counts):
    result = {}
    for c in COLORS:
        result[c] = 5
        for r in range(obs['fireworks'][c] + 1, 6):
            if counts[c, r] <= 0:
                result[c] = r - 1
                break
    return result


def support(card, counts, promises):
    options = {(c, r) for c in card['possible_colors']
               for r in card['possible_ranks'] if counts.get((c, r), 0) > 0}
    if card['card_id'] in promises:
        options &= promises[card['card_id']]
    return options


def exact_marginals(supports, counts):
    """Coefficient DP and reverse differentiation of a multiset partition function.
    A subset of labelled slots assigned k copies of a face has weight (n)_k.
    This conditions the whole hand jointly, not five independent draws.
    """
    n = len(supports)
    size = 1 << n
    full = size - 1
    layers = [[0.0] * size]
    layers[0][0] = 1.0
    transitions = []
    for face in FACES:
        available = counts.get(face, 0)
        eligible = sum(1 << i for i, values in enumerate(supports) if face in values)
        choices = [(0, 1.0)]
        subset = eligible
        while subset:
            k = subset.bit_count()
            if k <= available:
                weight = 1.0
                for j in range(k):
                    weight *= available - j
                choices.append((subset, weight))
            subset = (subset - 1) & eligible
        transitions.append(choices)
        old = layers[-1]
        new = [0.0] * size
        for mask, value in enumerate(old):
            if value:
                for subset, weight in choices:
                    if not mask & subset:
                        new[mask | subset] += value * weight
        layers.append(new)
    total = layers[-1][full]
    if not total:
        return None
    marginals = [dict() for _ in range(n)]
    adjoint = [0.0] * size
    adjoint[full] = 1.0
    for t in range(len(FACES) - 1, -1, -1):
        prev = [0.0] * size
        for mask, value in enumerate(layers[t]):
            for subset, weight in transitions[t]:
                if mask & subset:
                    continue
                derivative = adjoint[mask | subset] * weight
                prev[mask] += derivative
                mass = value * derivative / total
                if mass and subset:
                    for i in range(n):
                        if subset & (1 << i):
                            marginals[i][FACES[t]] = marginals[i].get(FACES[t], 0.0) + mass
        adjoint = prev
    return marginals


def dead(face, obs, ceiling):
    c, r = face
    return r <= obs['fireworks'][c] or r > ceiling[c]


def playable(face, obs):
    return face[1] == obs['fireworks'][face[0]] + 1


def focus(hand, action, obs, promises):
    """Public, deterministic decoder. Never reads a card's visible face."""
    counts = public_counts(obs)
    ceiling = ceilings(obs, counts)
    attr = 'color' if action['type'] == 'REVEAL_COLOR' else 'rank'
    value = action[attr]
    # Caller supplies touched IDs: sender gets these from visible cards; receiver
    # gets precisely the same set from the public clue event.
    touched = set(action.get('touched_card_ids', []))
    candidates = []
    for card in hand:
        if card['card_id'] not in touched:
            continue
        s = support(card, counts, promises)
        if s and not all(dead(x, obs, ceiling) for x in s):
            candidates.append(card)
    if not candidates:
        return None
    # A known matching next rank takes precedence (rank setup then color clue).
    if attr == 'color':
        ready = [c for c in candidates if c['possible_ranks'] == [obs['fireworks'][value] + 1]]
        if ready:
            return ready[0]['card_id']
    fresh = [c for c in candidates if c['card_id'] not in promises]
    candidates = fresh or candidates
    return candidates[-1 if attr == 'color' else 0]['card_id']


def signal(action, before, promises):
    # At eight tokens every clue is literal: an always-legal neutral channel.
    # Otherwise colors promise newest focus; rank 1 promises oldest focus.
    # Ranks 2..5 are always literal/save clues.
    if before['information_tokens'] == 8:
        return None, set()
    if action['type'] == 'REVEAL_COLOR':
        c = action['color']
        promised = {(c, before['fireworks'][c] + 1)}
    elif action['type'] == 'REVEAL_RANK' and action['rank'] == 1:
        promised = {(c, 1) for c in COLORS if before['fireworks'][c] == 0}
    else:
        return None, set()
    target = action['target']
    return focus(before['hands'][target], action, before, promises), promised


def literal_after(hand, action, touched):
    result = []
    attr = 'color' if action['type'] == 'REVEAL_COLOR' else 'rank'
    key = 'possible_colors' if attr == 'color' else 'possible_ranks'
    for card in hand:
        copy = dict(card)
        if card['card_id'] in touched:
            copy[key] = [action[attr]]
            copy[attr + '_clued'] = True
        else:
            copy[key] = [x for x in card[key] if x != action[attr]]
        result.append(copy)
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.promises = {}  # Public protocol evidence only: id -> possible faces.
        self.sessions = {}  # Strictly separate private model transcripts.
        self.calls = 0
        self.last_call = -10

    def metrics(self, obs):
        counts = public_counts(obs)
        ceiling = ceilings(obs, counts)
        unseen = counts.copy()
        seat = obs['observer']
        visible = Counter()
        for p, hand in enumerate(obs['hands']):
            if p != seat:
                for card in hand:
                    x = card['color'], card['rank']
                    unseen[x] -= 1
                    visible[x] += 1
        hand = obs['hands'][seat]
        supports = [support(card, unseen, self.promises) for card in hand]
        probabilities = exact_marginals(supports, unseen)
        if probabilities is None:
            # Recovery stays private to this decision: never publish a deduction
            # based on this seat's visible-card exclusions into shared promises.
            self.ctx.log('belief_inconsistent', seat=seat, turn=obs['turn'])
            supports = [support(card, unseen, {}) for card in hand]
            probabilities = exact_marginals(supports, unseen)
        if probabilities is None:
            # An inconsistent literal observation is not evidence for a gamble.
            # Independent literal support is conservative for guaranteed plays.
            probabilities = [{x: unseen.get(x, 0) / max(1, sum(unseen.get(y, 0) for y in s))
                              for x in s} for s in supports]
        data = []
        for card, options, dist in zip(hand, supports, probabilities):
            pplay = sum(p for x, p in dist.items() if playable(x, obs))
            pdead = sum(p for x, p in dist.items() if dead(x, obs, ceiling))
            loss = 0.0
            critical = 0.0
            for x, p in dist.items():
                if dead(x, obs, ceiling):
                    continue
                c, r = x
                if counts[x] == 1:
                    critical += p
                    loss += p * (ceiling[c] - r + 1)
                else:
                    # Preserve low prerequisites, but prefer discarding duplicates
                    # whose other copy is currently visible and protected.
                    value = (0.45 + 0.10 * (ceiling[c] - r)) / counts[x]
                    if visible[x]:
                        value *= 0.3
                    loss += p * value
            data.append({'index': card['index'], 'id': card['card_id'],
                         'p_play': pplay, 'p_dead': pdead, 'p_critical': critical,
                         'loss': loss, 'safe': bool(options) and all(playable(x, obs) for x in dist),
                         'protected': card['rank_clued'] or card['card_id'] in self.promises,
                         'faces': sorted((c + str(r), round(p, 3)) for (c, r), p in dist.items())})
        return data, counts, ceiling

    def candidates(self, obs):
        data, counts, ceiling = self.metrics(obs)
        own = obs['hands'][obs['observer']]
        final = obs.get('final_turns_remaining')
        choices = []
        for action in obs['legal_actions']:
            kind = action['type']
            if kind == 'PLAY':
                m = data[action['card_index']]
                if m['safe']:
                    value = 5.0 + 0.1 * m['p_critical']
                    # Unlock visible next cards; evaluate actual revealed future
                    # only against this seat's legitimate own-hand distribution.
                    for name, p in m['faces']:
                        c, r = name[0], int(name[1:])
                        value += p * (0.8 if r == 5 else 0)
                        value += p * 0.25 * sum(card['color'] == c and card['rank'] == r + 1
                            for s, hand in enumerate(obs['hands']) if s != obs['observer'] for card in hand)
                    choices.append({'action': action, 'value': value, 'reason': 'guaranteed play'})
                elif final is not None and obs['life_tokens'] > 1 and m['p_play'] > 0.05:
                    choices.append({'action': action, 'value': 4.5 * m['p_play'], 'reason': 'final round risk; spare life'})
                elif final == 1 and obs['life_tokens'] == 1 and m['p_play'] > obs['score'] / (obs['score'] + 1.0):
                    choices.append({'action': action, 'value': m['p_play'] * (obs['score'] + 1) - obs['score'], 'reason': 'positive final expected score'})
            elif kind == 'DISCARD':
                m = data[action['card_index']]
                value = 1.15 + 1.3 * m['p_dead'] - 2.5 * m['loss']
                if m['protected'] and m['p_dead'] < 0.999:
                    value -= 0.4
                if obs['information_tokens'] == 0:
                    value += 1.5
                if final is not None:
                    value = -0.1 - m['loss'] * 0.05
                value += (len(own) - m['index']) * 0.006
                choices.append({'action': action, 'value': value, 'reason': 'discard: expected loss %.3f, dead %.2f' % (m['loss'], m['p_dead'])})
            else:
                target = action['target']
                hand = obs['hands'][target]
                attr = 'color' if kind == 'REVEAL_COLOR' else 'rank'
                touched = {card['card_id'] for card in hand if card[attr] == action[attr]}
                extended = dict(action, touched_card_ids=list(touched))
                fid, promised = signal(extended, obs, self.promises)
                if fid is not None:
                    card = next(c for c in hand if c['card_id'] == fid)
                    if (card['color'], card['rank']) not in promised:
                        continue  # Never offer Astra a false convention promise.
                elif obs['information_tokens'] < 8 and (kind == 'REVEAL_COLOR' or action.get('rank') == 1):
                    continue
                newpromises = dict(self.promises)
                if fid is not None:
                    newpromises[fid] = promised
                after = literal_after(hand, action, touched)
                before_ready = set()
                after_ready = set()
                information = 0.0
                save = 0.0
                for old, new in zip(hand, after):
                    s0 = support(old, counts, self.promises)
                    s1 = support(new, counts, newpromises)
                    if s0 and all(playable(x, obs) for x in s0):
                        before_ready.add(old['card_id'])
                    if s1 and all(playable(x, obs) for x in s1):
                        after_ready.add(old['card_id'])
                    if s0:
                        information += (len(s0) - len(s1)) / len(s0)
                    face = old['color'], old['rank']
                    if old['card_id'] in touched and not old['rank_clued'] and not dead(face, obs, ceiling) and counts[face] == 1:
                        # Saving a threatened old unique prerequisite is worth
                        # more than saving a fresh five far from being discarded.
                        urgency = 1.0 if old['index'] <= 1 else 0.25
                        save += urgency * (0.7 + 0.25 * (ceiling[face[0]] - face[1]))
                new_ready = after_ready - before_ready
                distinct = {(c['color'], c['rank']) for c in hand if c['card_id'] in new_ready}
                value = -0.6 + 2.9 * bool(new_ready) + 1.1 * max(0, len(distinct) - 1)
                value += 0.16 * information + save
                if before_ready:
                    value -= 1.0
                if obs['information_tokens'] <= 1:
                    value -= 0.65
                if obs['information_tokens'] >= 6:
                    value += 0.45
                if final is not None:
                    distance = (target - obs['observer']) % obs['num_players']
                    if distance >= final:
                        value = -1.0
                    else:
                        value = 3.0 if new_ready and not before_ready else -0.5
                choices.append({'action': action, 'value': value,
                                'reason': 'clue: new playable IDs %s, save %.2f, focus %s' % (sorted(new_ready), save, fid)})
        choices.sort(key=lambda c: c['value'], reverse=True)
        return choices, data, ceiling

    def consult(self, obs, choices, data, ceiling):
        best = choices[0]
        budget = self.ctx.budget.remaining()
        # Six short deliberations leave ample deadline margin for a complete game.
        # Forced moves and a single clearly superior action need no model call.
        if (self.calls >= 6 or obs['turn'] - self.last_call < 7 or len(choices) < 2
                or budget.get('wall_s', 0) < 170 or budget.get('system2_calls', 0) <= 1
                or best['value'] - choices[1]['value'] > 1.3
                or best['reason'] == 'guaranteed play'):
            return best
        offered = [c for c in choices[:7] if c['value'] >= best['value'] - 1.4]
        if len(offered) < 2:
            return best
        seat = obs['observer']
        payload = {'seat': seat, 'turn': obs['turn'], 'stacks': obs['fireworks'],
                   'tokens': obs['information_tokens'], 'lives': obs['life_tokens'],
                   'deck': obs['deck_size'], 'final_actions': obs['final_turns_remaining'],
                   'reachable_heights': ceiling, 'own_beliefs': data,
                   'visible_partner': obs['hands'][1-seat] if obs['num_players'] == 2 else
                       [h for p, h in enumerate(obs['hands']) if p != seat],
                   'recent_public_events': obs['history'][-8:],
                   'options': [{'option': i, **c} for i, c in enumerate(offered)]}
        prompt = ('You play standard two-seat Hanabi. Maximize terminal team score; bomb-out scores zero. '
                  'Code supplies legal and protocol-safe actions and exact joint multiplicity beliefs. '
                  'Choose a strategic tradeoff among the numbered options. Your private session belongs only to seat '
                  + str(seat) + '. Do not infer own faces from another seat. '
                  'Fixed shared protocol: at 8 tokens every clue is literal. Below 8, a color clue promises its '
                  'public focus is playable now (known next rank first, else newest touched without a previous promise); '
                  'rank-1 clues promise oldest unpromised nondead touched card is playable; ranks 2..5 are literal saves. '
                  'Code computes these promises, future safe plays, dead cards and critical copies. '
                  'A saved card is protected but can be released when dead. Clues cost turns/tokens; discards draw. '
                  'Prefer finishing prerequisites, productive multi-card clues, preserving last copies and avoiding '
                  'saving many distant fives at the expense of tempo. No invented finesse. '
                  'Reply ONLY with {"option":integer}; no explanation.\n' + json.dumps(payload, separators=(',', ':')))
        self.calls += 1
        self.last_call = obs['turn']
        try:
            answer = self.ctx.think(prompt, session_id=self.sessions.get(seat), timeout_s=min(120.0, budget['wall_s'] - 35))
            self.sessions[seat] = answer.session_id
            match = re.search(r'"option"\s*:\s*(\d+)', answer.text)
            if not match:
                match = re.fullmatch(r'\s*(\d+)\s*', answer.text)
            if match and 0 <= int(match.group(1)) < len(offered):
                return offered[int(match.group(1))]
        except Exception as exc:
            self.ctx.log('advice_unavailable', error=type(exc).__name__)
        return best

    def decode(self, before, after):
        """Only public events can change shared protocol knowledge."""
        events = [e for e in after.get('history', []) if e['turn'] >= before['turn']]
        for event in events:
            if event['type'].startswith('REVEAL_'):
                fid, values = signal(event, before, self.promises)
                if fid is not None and values:
                    self.promises[fid] = values
            elif event['type'] in ('PLAY', 'DISCARD'):
                self.promises.pop(event.get('card_id'), None)
        alive = {c['card_id'] for hand in after['hands'] for c in hand}
        self.promises = {cid: values for cid, values in self.promises.items() if cid in alive}

    def run_episode(self, game):
        self.promises = {}
        self.sessions = {}
        self.calls = 0
        self.last_call = -10
        obs = self.ctx.observe()
        failures = 0
        for _ in range(240):
            if self.ctx.done or obs.get('done') or not obs.get('legal_actions'):
                return
            choices, data, ceiling = self.candidates(obs)
            if not choices:
                # A rank clue always exists at full tokens. Otherwise discarding
                # is legal. This guard is for malformed stand-in observations.
                legal = obs['legal_actions']
                action = next((a for a in legal if a['type'] == 'DISCARD'),
                              next((a for a in legal if a['type'] == 'REVEAL_RANK'), legal[0]))
            else:
                chosen = self.consult(obs, choices, data, ceiling)
                action = chosen['action']
                self.ctx.log('policy', seat=obs['observer'], turn=obs['turn'], action=action,
                             reason=chosen['reason'], value=round(chosen['value'], 3))
            if self.ctx.done:
                return
            try:
                after = self.ctx.act(action)
            except Exception as exc:
                failures += 1
                if self.ctx.done or failures >= 2:
                    return
                obs = self.ctx.observe()
                continue
            if after is None:
                after = self.ctx.observe()
            if after.get('turn') == obs.get('turn') and not after.get('done'):
                failures += 1
                if failures >= 2:
                    return
            else:
                failures = 0
                self.decode(obs, after)
            obs = after


def build(ctx):
    return Harness(ctx)
