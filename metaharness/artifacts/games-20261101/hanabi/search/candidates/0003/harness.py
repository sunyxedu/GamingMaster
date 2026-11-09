"""Public rotating instruction codebook; exact seat-local card beliefs.

Rank clues command a safe play; color clues recommend a discard. The slot is
(attribute ordinal + floor(clue turn / 2)) modulo recipient hand length.
These are preagreed action conventions, NOT literal meanings of the clue.
No private observation, inferred face, or model session crosses seats.
"""
from __future__ import annotations

import json
import re

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
                chosen = entries[0]['action']
                reason = entries[0]['why']
                # Model only arbitrates non-forced choices with meaningful regret.
                # At most six short calls; automatic policy always completes the game.
                mem = private[seat]
                close = [e for e in entries if e['value'] >= entries[0]['value'] - 1.5][:6]
                uncertain = (len(close) > 1 and entries[0]['value'] < 6.0
                             and not obs.get('final_turns_remaining'))
                budget = ctx.budget.remaining()
                if (adviser_enabled and uncertain and calls < 6 and mem['calls'] < 3
                        and obs['turn'] - mem['last_call_turn'] >= 8
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
                        'are exact for literal masks but omit policy selection bias. Compare the options '
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


def build(ctx):
    return Harness(ctx)
