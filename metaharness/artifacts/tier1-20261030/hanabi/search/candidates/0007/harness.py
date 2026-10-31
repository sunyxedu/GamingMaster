"""Public instruction codebook with bounded posterior policy evaluation.

Rank clues command a safe play; color clues recommend a discard. The slot is
(attribute ordinal + floor(clue turn / 2)) modulo recipient hand length.
These are preagreed action conventions, NOT literal meanings of the clue.
No private observation, inferred face, or model session crosses seats.
Root alternatives are compared by sampled terminal scores with paired
uncertainty estimates. Every simulated policy sees a redacted observation.
"""
from __future__ import annotations

import json
import re
import random
import time

COLORS = ('R', 'Y', 'G', 'W', 'B')
FACES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = (3, 2, 2, 2, 1) * 5


def signal_slot(action, turn, size):
    ordinal = action['rank'] - 1 if action['type'] == 'REVEAL_RANK' else COLORS.index(action['color'])
    return (ordinal + turn // 2) % size


def accounting(obs):
    heights = obs['fireworks']
    remaining = list(COPIES)
    for k, (c, r) in enumerate(FACES):
        if r <= heights[c]:
            remaining[k] -= 1
    for card in obs['discard_pile']:
        remaining[FACES.index((card['color'], card['rank']))] -= 1
    ceiling = {}
    for c in COLORS:
        ceiling[c] = 5
        for r in range(heights[c] + 1, 6):
            if remaining[FACES.index((c, r))] <= 0:
                ceiling[c] = r - 1
                break
    unseen = remaining[:]
    for seat, hand in enumerate(obs['hands']):
        if seat != obs['observer']:
            for card in hand:
                unseen[FACES.index((card['color'], card['rank']))] -= 1
    return remaining, [max(0, n) for n in unseen], ceiling


def joint_marginals(hand, counts):
    """Exact copy-weighted marginals conditional on all literal masks.

    Subset DP distributes identities among at most five labelled hand slots.
    An identity assigned to k slots contributes (copies)_k, without replacement.
    This is a literal-information prior, not a model of policy selection bias.
    """
    n = len(hand)
    full = (1 << n) - 1
    options = []
    for face, copies in zip(FACES, counts):
        allowed = 0
        for j, card in enumerate(hand):
            if face[0] in card['possible_colors'] and face[1] in card['possible_ranks']:
                allowed |= 1 << j
        weights = [(0, 1)]
        subset = allowed
        while subset:
            k = subset.bit_count()
            if k <= copies:
                weight = 1
                for m in range(k):
                    weight *= copies - m
                weights.append((subset, weight))
            subset = (subset - 1) & allowed
        options.append(weights)

    def advance(previous, weights):
        result = [0] * (full + 1)
        for mask, value in enumerate(previous):
            if value:
                for subset, weight in weights:
                    if not mask & subset:
                        result[mask | subset] += value * weight
        return result

    start = [0] * (full + 1)
    start[0] = 1
    forward = [start]
    for weights in options:
        forward.append(advance(forward[-1], weights))
    backward = [None] * 26
    backward[25] = start
    for k in range(24, -1, -1):
        backward[k] = advance(backward[k + 1], options[k])
    total = forward[-1][full]
    result = [[0.0] * 25 for _ in hand]
    if not total:
        # A defensive, conservative literal fallback; no empty-set certainty.
        for j, card in enumerate(hand):
            allowed = [k for k, (c, r) in enumerate(FACES)
                       if c in card['possible_colors'] and r in card['possible_ranks']]
            for k in allowed:
                result[j][k] = 1.0 / len(allowed)
        return result
    for k, weights in enumerate(options):
        for subset, weight in weights:
            if not subset:
                continue
            rest = full ^ subset
            before = rest
            ways = 0
            while True:
                ways += forward[k][before] * backward[k + 1][rest ^ before]
                if not before:
                    break
                before = (before - 1) & rest
            probability = ways * weight / total
            for j in range(n):
                if subset & (1 << j):
                    result[j][k] += probability
    return result


def card_loss(face, remaining, ceiling, heights, deck):
    c, r = face
    if r <= heights[c] or r > ceiling[c]:
        return 0.0
    copies = remaining[FACES.index(face)]
    distance = r - heights[c]
    potential = (ceiling[c] - r + 1) * (0.84 ** (distance - 1))
    if copies <= 1:
        return potential
    # Spare copies can be buried or discarded later. Penalize low prerequisites.
    return potential * (0.15 if copies == 2 else 0.07) * (1.3 if deck < 10 else 1.0)


def features(obs):
    remaining, unseen, ceiling = accounting(obs)
    h = obs['fireworks']
    own = obs['hands'][obs['observer']]
    beliefs = joint_marginals(own, unseen)
    stats = []
    for card, belief in zip(own, beliefs):
        pplay = pdead = pcritical = loss = 0.0
        for k, ((c, r), probability) in enumerate(zip(FACES, belief)):
            if not probability:
                continue
            pplay += probability * (r == h[c] + 1)
            pdead += probability * (r <= h[c] or r > ceiling[c])
            pcritical += probability * (h[c] < r <= ceiling[c] and remaining[k] == 1)
            loss += probability * card_loss((c, r), remaining, ceiling, h, obs['deck_size'])
        stats.append(dict(index=card['index'], card_id=card['card_id'],
                          play=pplay, dead=pdead, critical=pcritical, loss=loss,
                          faces=[f'{c}{r}:{belief[k]:.2f}' for k, (c, r) in enumerate(FACES)
                                 if belief[k] > 0.07]))
    return remaining, ceiling, stats


def candidates(obs, remaining, ceiling, stats):
    """Validate every play promise against the giver's permitted visible hand."""
    seat = obs['observer']
    h = obs['fireworks']
    tokens = obs['information_tokens']
    deck = obs['deck_size']
    final = obs.get('final_turns_remaining')
    entries = []
    for action in obs['legal_actions']:
        kind = action['type']
        detail = ''
        value = -1000.0
        if kind == 'PLAY':
            s = stats[action['card_index']]
            if s['play'] > 1 - 1e-9:
                value = 10.0 + 0.3 * s['critical']
                detail = 'literal guaranteed play'
            elif final is not None:
                # Life loss cannot terminate with two lives; on the last life,
                # compare the one-point gain against losing the entire score.
                p = s['play']
                gain = p if obs['life_tokens'] > 1 else p - (1 - p) * (sum(h.values()) + 1)
                if gain > 0:
                    value = 2.0 + gain if final == 1 else gain
                    detail = f'final-round risk, p_success={p:.3f}'
        elif kind == 'DISCARD':
            s = stats[action['card_index']]
            value = 1.1 + 1.6 * s['dead'] - 3.5 * s['loss']
            if tokens <= 1:
                value += 2.0
            if final is not None:
                value = -1.0 - s['loss']
            detail = f'own discard E_loss={s["loss"]:.3f}, p_dead={s["dead"]:.3f}'
        elif kind in ('REVEAL_RANK', 'REVEAL_COLOR'):
            target = action['target']
            if target != (seat + 1) % obs['num_players']:
                continue
            hand = obs['hands'][target]
            if not hand:
                continue
            slot = signal_slot(action, obs['turn'], len(hand))
            card = hand[slot]
            face = (card['color'], card['rank'])
            loss = card_loss(face, remaining, ceiling, h, deck)
            if kind == 'REVEAL_RANK':
                if card['rank'] != h[card['color']] + 1:
                    continue  # Never transmit an unsafe play instruction.
                if final == 1:
                    continue
                value = 6.0 + 0.35 * (5 - card['rank'])
                if card['rank'] == 5:
                    value += 0.5
                # Favor prerequisites already followed by a visible next rank.
                if any(x['color'] == card['color'] and x['rank'] == card['rank'] + 1 for x in hand):
                    value += 0.5
                if final is not None:
                    value = 4.0
                detail = f'command PLAY {slot}, guaranteed {card["color"]}{card["rank"]}'
            else:
                # Every color clue is a discard recommendation, never a play.
                # Low-loss commands supply a known safe draw without guessing.
                value = 2.2 - 3.5 * loss
                if loss == 0:
                    value += 0.7
                if tokens <= 1:
                    value -= 2.0
                if card['rank'] == h[card['color']] + 1:
                    value -= 1.5
                if final is not None:
                    value = -2.0 - loss
                detail = f'command DISCARD {slot}, visible {card["color"]}{card["rank"]}, loss={loss:.3f}'
        if value > -999:
            entries.append(dict(action=action, value=value, why=detail))
    entries.sort(key=lambda e: e['value'], reverse=True)
    return entries


def fast_features(obs):
    """Conservative per-slot rollout beliefs; real decisions use exact DP."""
    remaining, unseen, ceiling = accounting(obs)
    h = obs['fireworks']
    stats = []
    for i, card in enumerate(obs['hands'][obs['observer']]):
        indices = [k for k, (c, r) in enumerate(FACES) if unseen[k] > 0
                   and c in card['possible_colors'] and r in card['possible_ranks']]
        total = sum(unseen[k] for k in indices)
        if not total:
            indices = [k for k, (c, r) in enumerate(FACES)
                       if c in card['possible_colors'] and r in card['possible_ranks']]
            weights = {k: 1 / max(1, len(indices)) for k in indices}
        else:
            weights = {k: unseen[k] / total for k in indices}
        stats.append(dict(index=i, card_id=card['card_id'],
            play=sum(p for k,p in weights.items() if FACES[k][1] == h[FACES[k][0]]+1),
            dead=sum(p for k,p in weights.items() if FACES[k][1] <= h[FACES[k][0]]
                     or FACES[k][1] > ceiling[FACES[k][0]]),
            critical=sum(p for k,p in weights.items() if remaining[k] == 1
                         and h[FACES[k][0]] < FACES[k][1] <= ceiling[FACES[k][0]]),
            loss=sum(p*card_loss(FACES[k],remaining,ceiling,h,obs['deck_size'])
                     for k,p in weights.items())))
    return remaining, ceiling, stats


def instructed(obs, entries):
    """The parent's public decoder, shared by real play and simulation."""
    incoming = obs.get('history', [])[-1:]
    hand = obs['hands'][obs['observer']]
    if not incoming or not hand:
        return None
    event = incoming[0]
    if (event.get('target') != obs['observer'] or event.get('turn') != obs['turn']-1
            or event.get('type') not in ('REVEAL_RANK', 'REVEAL_COLOR')):
        return None
    slot = signal_slot(event, event['turn'], len(hand))
    kind = 'PLAY' if event['type'] == 'REVEAL_RANK' else 'DISCARD'
    action = dict(type=kind, card_index=slot)
    if action not in obs['legal_actions']:
        return None
    if kind == 'DISCARD':
        scoring = [e for e in entries if e['action']['type'] in ('PLAY','REVEAL_RANK')]
        if scoring:
            return scoring[0]['action']
    return action


class RolloutWorld:
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
                    hands=hands,discard_pile=self.discards,history=self.history,
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

    def finish(self, action, stop):
        self.step(action)
        for _ in range(180):
            if self.done or stop():
                break
            obs = self.view()
            remaining, cap, stats = fast_features(obs)
            entries = candidates(obs,remaining,cap,stats)
            if not entries:
                break
            self.step(instructed(obs,entries) or entries[0]['action'])
        return float(sum(self.h.values()) if self.lives > 0 else 0)


def sample_world(obs, rng):
    """Exact sequential conditional sampling of the literal joint posterior."""
    _, counts, _ = accounting(obs)
    hand = obs['hands'][obs['observer']]
    faces = []
    for i in range(len(hand)):
        marginal = joint_marginals(hand[i:], counts)[0]
        total = sum(p for k,p in enumerate(marginal) if counts[k] > 0)
        if total <= 0:
            return None
        pick = rng.random()*total
        selected = None
        for k,p in enumerate(marginal):
            if counts[k] > 0:
                pick -= p
                if pick <= 0:
                    selected = k
                    break
        if selected is None:
            selected = max(k for k,p in enumerate(marginal) if p > 0 and counts[k] > 0)
        faces.append(FACES[selected])
        counts[selected] -= 1
    deck = [face for face,n in zip(FACES,counts) for _ in range(n)]
    if len(deck) != obs['deck_size']:
        return None
    rng.shuffle(deck)
    return faces, deck


def search_options(ctx, obs, entries):
    """Bounded common-random-world policy evaluation; no shared hidden memory."""
    if len(entries) <= 1 or obs['num_players'] != 2:
        return entries
    # Cover the best local options of every action type, rather than searching
    # only tiny perturbations of the local utility's preferred action.
    limits = {'PLAY':5,'DISCARD':5,'REVEAL_RANK':5,'REVEAL_COLOR':5}
    options = []
    for e in entries:
        kind = e['action']['type']
        if limits.get(kind,0) > 0:
            options.append(dict(e))
            limits[kind] -= 1
    if len(options) <= 1:
        return entries
    budget = ctx.budget.remaining()
    if budget.get('wall_s',0) < 30:
        return entries
    deadline = time.monotonic()+min(8.0,max(.1,budget.get('wall_s',0)-20))
    stop = lambda: ctx.done or time.monotonic() >= deadline
    # Fixed arbitrary sampling stream ensures repeatability, NOT deal prediction.
    rng = random.Random(73061+obs['turn']*1009+obs['observer'])
    totals = [0.0]*len(options)
    squares = [0.0]*len(options)
    delta_totals = [0.0]*len(options)
    delta_squares = [0.0]*len(options)
    n = 0
    for _ in range(96):
        if stop():
            break
        sampled = sample_world(obs,rng)
        if sampled is None:
            break
        values = []
        for e in options:
            if stop():
                break
            world = RolloutWorld(obs,*sampled)
            score = world.finish(e['action'],stop)
            if not world.done:
                break  # Never compare completed values with truncated ones.
            values.append(score)
        if len(values) != len(options):
            break
        n += 1
        for i,value in enumerate(values):
            totals[i] += value
            squares[i] += value*value
            delta = value-values[0]
            delta_totals[i] += delta
            delta_squares[i] += delta*delta
    if n < 8:
        return entries
    for i,e in enumerate(options):
        mean = totals[i]/n
        e['rollout_score'] = round(mean,4)
        e['rollout_n'] = n
        e['rollout_sd'] = round(max(0,squares[i]/n-mean*mean)**.5,3)
        delta = delta_totals[i]/n
        se = (max(0,delta_squares[i]/n-delta*delta)/(n-1))**.5
        e['paired_gain'] = round(delta,4)
        e['paired_se'] = round(se,4)
        # A one-sided confidence margin on paired score differences prevents
        # maximization over many noisy alternatives from causing gratuitous
        # deviations. Common worlds reduce variance relative to unpaired SDs.
        e['search_value'] = delta - 1.25*se - .10 if i else 0.0
    options.sort(key=lambda e:(e['search_value'],e['value']),reverse=True)
    ctx.log('rollout_comparison',turn=obs['turn'],samples=n,
            options=[dict(action=e['action'],score=e['rollout_score'],sd=e['rollout_sd'],
                          gain=e['paired_gain'],paired_se=e['paired_se'],
                          evidence=round(e['search_value'],4),
                          local_value=round(e['value'],3)) for e in options])
    return options


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        # Fresh even if a caller reuses the Harness object across resets.
        private = {i: {'session': None, 'calls': 0, 'last_call_turn': -100}
                   for i in range(5)}
        calls = 0
        adviser_enabled = True
        failures = 0
        obs = ctx.observe()
        for _ in range(240):
            if ctx.done or not isinstance(obs, dict) or obs.get('done'):
                return
            legal = obs.get('legal_actions', [])
            if not legal:
                return
            seat = obs['observer']
            hand = obs['hands'][seat]
            remaining, ceiling, stats = features(obs)
            entries = candidates(obs, remaining, ceiling, stats)
            chosen = None
            reason = ''
            incoming = obs.get('history', [])[-1:]
            if incoming:
                event = incoming[0]
                if (event.get('target') == seat and event.get('turn') == obs['turn'] - 1
                        and event.get('type') in ('REVEAL_RANK', 'REVEAL_COLOR') and hand):
                    slot = signal_slot(event, event['turn'], len(hand))
                    kind = 'PLAY' if event['type'] == 'REVEAL_RANK' else 'DISCARD'
                    proposed = {'type': kind, 'card_index': slot}
                    if kind == 'PLAY' and proposed in legal:
                        chosen, reason = proposed, 'decode public rank play instruction'
                    elif kind == 'DISCARD' and proposed in legal:
                        # Recommendations yield to scoring on EITHER side. Otherwise
                        # repeated commands can leave the giver's initial hand stranded.
                        scoring = [e for e in entries if e['action']['type'] in ('PLAY', 'REVEAL_RANK')]
                        if scoring:
                            chosen = scoring[0]['action']
                            reason = 'scoring supersedes discard recommendation: ' + scoring[0]['why']
                        else:
                            chosen, reason = proposed, 'decode public color discard instruction'
            if chosen is None:
                if not entries:
                    return
                entries = search_options(ctx, obs, entries)
                chosen = entries[0]['action']
                reason = entries[0]['why']
                if 'rollout_score' in entries[0]:
                    reason = ('rollout E_terminal=' + str(entries[0]['rollout_score'])
                              + ': ' + reason)
                # Model only arbitrates non-forced choices with meaningful regret.
                # At most six short calls; automatic policy always completes the game.
                mem = private[seat]
                if 'rollout_score' in entries[0]:
                    threshold = max(0.0,entries[0]['search_value']-.2)
                    close = [e for e in entries if e['search_value'] >= threshold][:5]
                else:
                    close = [e for e in entries if e['value'] >= entries[0]['value'] - 1.5][:5]
                uncertain = (len(close) > 1 and entries[0]['value'] < 6.0
                             and not obs.get('final_turns_remaining'))
                budget = ctx.budget.remaining()
                if (adviser_enabled and uncertain and calls < 2 and mem['calls'] < 1
                        and obs['turn'] - mem['last_call_turn'] >= 16
                        and budget.get('system2_calls', 0) > 1
                        and budget.get('wall_s', 0) > 150):
                    view = dict(seat=seat, turn=obs['turn'], fireworks=obs['fireworks'],
                                tokens=obs['information_tokens'], lives=obs['life_tokens'],
                                deck=obs['deck_size'], ceiling=ceiling,
                                own_beliefs=stats,
                                visible_hands={str(i): cards for i, cards in enumerate(obs['hands']) if i != seat},
                                recent_public=obs.get('history', [])[-6:],
                                options=[dict(id=i, **e) for i, e in enumerate(close)])
                    prompt = (
                        'Choose a Hanabi action to maximize terminal score 0-25. '
                        'You see only this seat\'s permitted information. Three lost lives scores zero. '
                        'Codebook is fixed: rank clues command an immediate safe play at the supplied slot; '
                        'color clues recommend discarding the supplied slot. The encoded slot need not '
                        'be touched by the clue. All listed play commands are verified safe by the giver. '
                        'Own scoring plays and verified safe partner play clues override color discard recommendations. '
                        'Do not invent clues or conventions. Tokens are scarce: own discards restore one; '
                        'a color clue plus recipient discard only breaks even. Avoid losing unique '
                        'prerequisites; favor draws from dead or duplicate cards. Belief probabilities '
                        'are exact for literal masks but omit policy selection bias. Rollout_score is '
                        'sampled terminal score under this protocol, from correctly redacted future '
                        'seat views. It is a noisy policy estimate, not an omniscient oracle. Favor '
                        'the largest rollout score unless a concrete strategic defect explains it. Compare the options '
                        'and return only JSON {"choice": INTEGER}. No explanation.\n' + json.dumps(view))
                    calls += 1
                    mem['calls'] += 1
                    mem['last_call_turn'] = obs['turn']
                    try:
                        # Backend timeout excludes some service queueing. Bound the
                        # caller's wait as well; ctx owns/cancels this model task.
                        future = ctx.spawn(lambda p=prompt, s=mem['session']:
                                           ctx.think(p, session_id=s, timeout_s=90.0))
                        reply = future.result(timeout=90.0)
                        mem['session'] = reply.session_id
                        match = re.search(r'"choice"\s*:\s*(\d+)', reply.text)
                        if match and 0 <= int(match.group(1)) < len(close):
                            option = close[int(match.group(1))]
                            chosen, reason = option['action'], 'Astra: ' + option['why']
                    except Exception as exc:
                        # Budget/episode closure, timeout and backend failure are handled.
                        # At most one abandoned model task; finish deterministically.
                        adviser_enabled = False
                        ctx.log('adviser_failed', error=type(exc).__name__)
                        if ctx.done:
                            return
            if chosen not in legal:
                return
            ctx.log('decision', seat=seat, turn=obs['turn'], action=chosen, reason=reason)
            old_turn = obs['turn']
            try:
                result = ctx.act(chosen)
            except Exception as exc:
                if ctx.done:
                    return
                failures += 1
                if failures >= 2:
                    return
                obs = ctx.observe()
                continue
            obs = result if isinstance(result, dict) else ctx.observe()
            if obs.get('turn') == old_turn and not obs.get('done'):
                failures += 1
                if failures >= 2:
                    return
            else:
                failures = 0
                event = obs.get('history', [])[-1:]
                if event and event[0].get('type') == 'DISCARD':
                    face = (event[0]['color'],event[0]['rank'])
                    # 'remaining' and 'ceiling' are from the preceding actor's
                    # observation, before the now-public discarded face left.
                    if (remaining[FACES.index(face)] == 1
                            and face[1] <= ceiling[face[0]]
                            and face[1] > obs['fireworks'][face[0]]):
                        ctx.log('lost_live_last_copy',turn=old_turn,face=face[0]+str(face[1]),
                                ceiling_points=ceiling[face[0]]-face[1]+1)


def build(ctx):
    return Harness(ctx)
