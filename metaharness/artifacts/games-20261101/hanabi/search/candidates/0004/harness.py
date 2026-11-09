"""Adaptive public instruction channel with persistent noncritical certificates.

Only public domains cross seats. All actual visible faces are used locally to
validate messages; decoding never reads them. No engine or seed access.
"""
from __future__ import annotations

import json
import re

COLORS = ('R', 'Y', 'G', 'W', 'B')
FACES = tuple((c, r) for c in COLORS for r in range(1, 6))
COPIES = (3, 2, 2, 2, 1) * 5


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


def joint_marginals(hand, counts, domains=None):
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
            if (face[0] in card['possible_colors'] and face[1] in card['possible_ranks']
                    and (domains is None or face in domains[j])):
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
                       if c in card['possible_colors'] and r in card['possible_ranks']
                       and (domains is None or (c, r) in domains[j])]
            if not allowed:
                allowed = list(range(25))
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


def domains_for(hand, remaining, certificates):
    domains = []
    for card in hand:
        dom = {face for k, face in enumerate(FACES)
               if remaining[k] > 0 and face[0] in card['possible_colors']
               and face[1] in card['possible_ranks']}
        if card['card_id'] in certificates:
            dom &= certificates[card['card_id']]
        domains.append(dom)
    return domains


def public_safe(domains, heights):
    return [i for i, dom in enumerate(domains)
            if dom and all(r == heights[c] + 1 for c, r in dom)]


def decode(action, turn, hand, heights, remaining, ceiling, certificates,
           tokens_before):
    """Face-blind decoder. hand contains post-clue masks, never uses faces."""
    domains = domains_for(hand, remaining, certificates)
    safe = public_safe(domains, heights)
    if safe or tokens_before == 8:
        return dict(mode='literal', safe=safe, slot=None, proof=None)
    kind = action['type']
    if kind == 'REVEAL_RANK':
        allowed = {face for face in FACES if face[1] == heights[face[0]] + 1}
        ordinal = action['rank'] - 1
        mode = 'play'
    else:
        # This is a snapshot restriction, not a permanent discard permission.
        allowed = {face for k, face in enumerate(FACES)
                   if (face[1] <= heights[face[0]] or face[1] > ceiling[face[0]]
                       or remaining[k] >= 2)}
        ordinal = COLORS.index(action['color'])
        mode = 'certificate'
    pool = [i for i, dom in enumerate(domains) if dom & allowed]
    if not pool:
        return dict(mode='literal', safe=[], slot=None, proof=None)
    slot = pool[(ordinal + turn // 2) % len(pool)]
    return dict(mode=mode, safe=[], slot=slot, proof=domains[slot] & allowed)


def clue_view(hand, action):
    """Sender applies an actual legal clue to public masks for decoder testing."""
    attr = 'rank' if action['type'] == 'REVEAL_RANK' else 'color'
    key = 'possible_ranks' if attr == 'rank' else 'possible_colors'
    value = action[attr]
    result = []
    for card in hand:
        copy = dict(card)
        touched = card[attr] == value
        copy[key] = [v for v in card[key] if (v == value) == touched]
        if touched:
            copy[attr + '_clued'] = True
        # The decoder is deliberately denied the sender's visible faces.
        copy['color'] = copy['rank'] = None
        result.append(copy)
    return result


def features(obs, certificates):
    remaining, unseen, ceiling = accounting(obs)
    heights = obs['fireworks']
    own = obs['hands'][obs['observer']]
    domains = domains_for(own, remaining, certificates)
    beliefs = joint_marginals(own, unseen, domains)
    stats = []
    visible = {(x['color'], x['rank']) for seat, hand in enumerate(obs['hands'])
               if seat != obs['observer'] for x in hand}
    for card, belief in zip(own, beliefs):
        pplay = pdead = pcritical = loss = five = unlock = 0.0
        for k, ((c, r), probability) in enumerate(zip(FACES, belief)):
            if not probability:
                continue
            pplay += probability * (r == heights[c] + 1)
            pdead += probability * (r <= heights[c] or r > ceiling[c])
            pcritical += probability * (heights[c] < r <= ceiling[c] and remaining[k] == 1)
            cost = card_loss((c, r), remaining, ceiling, heights, obs['deck_size'])
            # A visible spare is accessible; avoid sacrificing a hidden last copy.
            if remaining[k] > 1 and (c, r) in visible:
                cost *= 0.3
            loss += probability * cost
            five += probability * (r == 5)
            unlock += probability * ((c, r + 1) in visible)
        stats.append(dict(index=card['index'], card_id=card['card_id'],
                          play=pplay, dead=pdead, critical=pcritical, loss=loss,
                          five=five, unlock=unlock,
                          faces=[f'{c}{r}:{belief[k]:.2f}' for k, (c, r) in enumerate(FACES)
                                 if belief[k] > 0.07]))
    return remaining, ceiling, stats


def candidates(obs, remaining, ceiling, stats, certificates):
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
        metadata = {}
        if kind == 'PLAY':
            s = stats[action['card_index']]
            if s['play'] > 1 - 1e-9:
                value = 10.0 + 0.3 * s['critical'] + 0.4 * s['unlock'] + 0.2 * s['five']
                detail = 'guaranteed play from literal/public-certificate beliefs'
            elif final is not None:
                p = s['play']
                gain = p if obs['life_tokens'] > 1 else p - (1 - p) * sum(h.values())
                if gain > 0:
                    value = 2.0 + gain if final == 1 else gain
                    detail = f'final-round risk, p_success={p:.3f}'
        elif kind == 'DISCARD':
            s = stats[action['card_index']]
            # Dead identities already contribute zero to expected loss. An
            # extra dead-probability bonus double-counts that upside and can
            # favor a risky unknown over a certified redundant card.
            value = 1.1 - 3.5 * s['loss']
            if tokens <= 1:
                value += 2.0
            if final is not None:
                value = -1.0 - s['loss']
            detail = f'own discard E_loss={s["loss"]:.3f}, p_critical={s["critical"]:.3f}'
        elif kind in ('REVEAL_RANK', 'REVEAL_COLOR'):
            target = action['target']
            if target != (seat + 1) % obs['num_players'] or final == 1:
                continue
            hand = obs['hands'][target]
            if not hand:
                continue
            after = clue_view(hand, action)
            message = decode(action, obs['turn'], after, h, remaining, ceiling,
                             certificates, tokens)
            before_domains = domains_for(hand, remaining, certificates)
            before_safe = public_safe(before_domains, h)
            new_safe = [i for i in message['safe'] if i not in before_safe]
            unique = len({(hand[i]['color'], hand[i]['rank']) for i in new_safe})
            mode = message['mode']
            metadata = dict(mode=mode, slot=message['slot'], new_safe=new_safe)
            if mode == 'play':
                slot = message['slot']
                card = hand[slot]
                if (card['color'], card['rank']) not in message['proof']:
                    continue
                value = 6.0 + 0.35 * (5 - card['rank'])
                if card['rank'] == 5:
                    value += 0.5
                if any(x['color'] == card['color'] and x['rank'] == card['rank'] + 1 for x in hand):
                    value += 0.5
                if final is not None:
                    value = 4.0
                detail = f'adaptive PLAY {slot}, verified {card["color"]}{card["rank"]}'
            elif mode == 'certificate':
                slot = message['slot']
                card = hand[slot]
                face = (card['color'], card['rank'])
                if face not in message['proof']:
                    continue
                loss = card_loss(face, remaining, ceiling, h, deck)
                value = 2.2 - 3.5 * loss
                if loss == 0:
                    value += 0.7
                if tokens <= 1:
                    value -= 2.0
                if card['rank'] == h[card['color']] + 1:
                    value -= 1.5
                if final is not None:
                    value = -2.0 - loss
                detail = f'noncritical certificate slot {slot}, {card["color"]}{card["rank"]}, loss={loss:.3f}'
            else:
                # A literal 1 clue may unlock several colors, with duplicates
                # rechecked by exact beliefs after each intervening play.
                value = 6.5 + 1.1 * (unique - 1) if unique else -0.8
                if before_safe:
                    value -= 2.0
                if unique:
                    value += 0.15 * sum(5 - hand[i]['rank'] for i in new_safe)
                # Literal clues at full tokens need a sensible neutral choice.
                information = sum(len(a) - len(b) for a, b in zip(
                    before_domains, domains_for(after, remaining, certificates)))
                value += min(0.5, information * 0.01)
                if final is not None:
                    value = 4.0 if unique and not before_safe else -3.0
                detail = f'literal packet, {unique} newly playable colors, slots={new_safe}'
        if value > -999:
            entries.append(dict(action=action, value=value, why=detail, **metadata))
    entries.sort(key=lambda e: e['value'], reverse=True)
    return entries


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        private = {i: {'session': None, 'calls': 0, 'last_call_turn': -100}
                   for i in range(5)}
        # Every member is reconstructed from public actions/masks alone.
        certificates = {}
        processed_turn = -1
        calls = failures = 0
        adviser_enabled = True
        obs = ctx.observe()
        for _ in range(240):
            if ctx.done or not isinstance(obs, dict) or obs.get('done'):
                return
            legal = obs.get('legal_actions', [])
            if not legal:
                return
            seat = obs['observer']
            hand = obs['hands'][seat]
            remaining, _, ceiling = accounting(obs)
            incoming = None
            history = obs.get('history', [])
            if history:
                event = history[-1]
                if event['turn'] > processed_turn:
                    processed_turn = event['turn']
                    if (event.get('target') == seat and event['turn'] == obs['turn'] - 1
                            and event['type'] in ('REVEAL_RANK', 'REVEAL_COLOR')):
                        incoming = decode(event, event['turn'], hand, obs['fireworks'],
                                          remaining, ceiling, certificates,
                                          obs['information_tokens'] + 1)
                        if incoming['proof']:
                            cid = hand[incoming['slot']]['card_id']
                            certificates[cid] = incoming['proof']
                        ctx.log('public_message', seat=seat, turn=obs['turn'],
                                mode=incoming['mode'], slot=incoming['slot'],
                                domain=sorted(incoming['proof'] or []))
            active = {card['card_id'] for cards in obs['hands'] for card in cards}
            certificates = {cid: dom for cid, dom in certificates.items() if cid in active}
            remaining, ceiling, stats = features(obs, certificates)
            entries = candidates(obs, remaining, ceiling, stats, certificates)
            if not entries:
                return
            chosen = entries[0]['action']
            reason = entries[0]['why']
            forced = False
            if incoming and incoming['mode'] == 'play':
                action = {'type': 'PLAY', 'card_index': incoming['slot']}
                if action in legal:
                    chosen, reason, forced = action, 'execute adaptive rank instruction', True
            elif incoming and incoming['mode'] == 'certificate':
                # A certificate is durable information. Its immediate discard
                # recommendation yields to scoring in either seat.
                scoring = [e for e in entries if e['action']['type'] == 'PLAY'
                           or (e['action']['type'].startswith('REVEAL') and e['value'] >= 4.0)]
                action = {'type': 'DISCARD', 'card_index': incoming['slot']}
                if scoring:
                    chosen, reason = scoring[0]['action'], 'scoring over certificate: ' + scoring[0]['why']
                    forced = True
                elif action in legal:
                    chosen, reason, forced = action, 'execute certified noncritical discard', True
            mem = private[seat]
            close = [e for e in entries if e['value'] >= entries[0]['value'] - 1.2][:6]
            uncertain = (not forced and len(close) > 1 and entries[0]['value'] < 6.0
                         and obs.get('final_turns_remaining') is None)
            budget = ctx.budget.remaining()
            if (adviser_enabled and uncertain and calls < 3 and mem['calls'] < 2
                    and obs['turn'] - mem['last_call_turn'] >= 10
                    and budget.get('system2_calls', 0) > 1
                    and budget.get('wall_s', 0) > 180):
                view = dict(seat=seat, turn=obs['turn'], fireworks=obs['fireworks'],
                            tokens=obs['information_tokens'], lives=obs['life_tokens'],
                            deck=obs['deck_size'], ceiling=ceiling, own_beliefs=stats,
                            visible_hands={str(i): cards for i, cards in enumerate(obs['hands']) if i != seat},
                            recent_public=obs.get('history', [])[-6:],
                            options=[dict(id=i, **e) for i, e in enumerate(close)])
                prompt = (
                    'Choose an option for two-player Hanabi to maximize terminal score 0-25. '
                    'Three lost lives scores zero. Only this seat\'s permitted information is shown. '
                    'The protocol is fixed: literal packets let the recipient deduce safe plays; '
                    'adaptive rank messages command the supplied safe slot; color certificates '
                    'guarantee the supplied slot is currently noncritical and recommend discarding it. '
                    'Scoring actions override discard recommendations. Public certificates persist '
                    'as face-domain restrictions, not permanent permission to discard. All messages '
                    'are prevalidated. Preserve unique prerequisites, use clues efficiently, and '
                    'avoid drawing past useful held cards late in the deck. The supplied values '
                    'are heuristic; choose only a listed option. Return only {"choice": INTEGER}.\n'
                    + json.dumps(view, separators=(',', ':')))
                calls += 1
                mem['calls'] += 1
                mem['last_call_turn'] = obs['turn']
                try:
                    future = ctx.spawn(lambda p=prompt, s=mem['session']:
                                       ctx.think(p, session_id=s, timeout_s=75.0))
                    reply = future.result(timeout=80.0)
                    mem['session'] = reply.session_id
                    match = re.search(r'"choice"\s*:\s*(\d+)', reply.text)
                    if match and 0 <= int(match.group(1)) < len(close):
                        option = close[int(match.group(1))]
                        chosen, reason = option['action'], 'Astra: ' + option['why']
                except Exception as exc:
                    adviser_enabled = False
                    ctx.log('adviser_failed', error=type(exc).__name__)
                    if ctx.done:
                        return
            if chosen not in legal:
                return
            ctx.log('decision', seat=seat, turn=obs['turn'], action=chosen, reason=reason,
                    certificate_count=len(certificates))
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
