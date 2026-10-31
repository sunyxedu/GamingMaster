"""Isolated synchronous pyminion runner; newline JSON is the only stdout output."""
from __future__ import annotations

import json
import logging
import random
import sys
import traceback
from collections import Counter

from pyminion.bots.bot import BotDecider
from pyminion.bots.examples.big_money_ultimate import BigMoneyUltimateDecider
from pyminion.core import CardType
from pyminion.expansions import base as b
from pyminion.game import Game
from pyminion.player import Player

from engine_patches import apply
from scenarios import ADAPTOR_VERSION, ENGINE_COMMIT, KINGDOMS

CARDS = {c.name: c for c in b.base_set + [b.copper, b.silver, b.gold, b.estate, b.duchy, b.province, b.curse]}


def counts(cards):
    return dict(sorted(Counter(c.name for c in cards).items()))


def vp(owned):
    return (owned.get('Estate', 0) + 3 * owned.get('Duchy', 0)
            + 6 * owned.get('Province', 0) - owned.get('Curse', 0)
            + owned.get('Gardens', 0) * (sum(owned.values()) // 10))


class PublicPlayer(Player):
    """Track public ownership independently of hidden zones, including temporary cards."""
    def __init__(self, decider, player_id, events):
        super().__init__(decider, player_id=player_id)
        self.events = events
        self.owned = Counter(Copper=7, Estate=3)

    def emit(self, kind, card):
        self.events.append({'player': self.player_id, 'event': kind, 'card': card.name})

    def buy(self, card, game):
        super().buy(card, game)
        self.owned[card.name] += 1
        self.emit('buy', card)

    def gain(self, card, game, destination=None, source=None):
        super().gain(card, game, destination, source)
        self.owned[card.name] += 1
        self.emit('gain_to_hand' if destination is self.hand else 'gain', card)

    def trash(self, target_card, game, source=None):
        super().trash(target_card, game, source)
        self.owned[target_card.name] -= 1
        self.emit('trash', target_card)

    def play(self, target_card, game, generic_play=True):
        self.emit('play', target_card)
        super().play(target_card, game, generic_play)

    def exact_play(self, card, game, generic_play=True):
        self.emit('play', card)
        super().exact_play(card, game, generic_play)

    def discard(self, game, target_card, source=None, silent=False):
        super().discard(game, target_card, source, silent)
        if not silent:
            self.emit('discard', target_card)

    def reveal(self, cards, game, message=None):
        cards = [cards] if hasattr(cards, 'name') else list(cards)
        super().reveal(cards, game, message)
        for card in cards:
            self.emit('reveal', card)


class MoneyPolicy(BotDecider):
    """BM-1, SM-1 and WM-1, exactly the study's versioned policies."""
    def __init__(self, policy):
        self.policy = policy
        self.target = {'SM-1': b.smithy, 'WM-1': b.witch}.get(policy)

    def action_phase_decision(self, valid_actions, player, game):
        return self.target if self.target in valid_actions else None

    def buy_phase_decision(self, valid_cards, player, game):
        n = game.supply.pile_length('Province')
        priority = [b.province]
        if n <= 4:
            priority.append(b.duchy)
        if n <= 2:
            priority.append(b.estate)
        if self.target and n >= 5 and player.get_card_count(self.target) == 0:
            if self.policy != 'WM-1' or game.supply.pile_length('Curse') > 0:
                priority.append(self.target)
        priority += [b.gold, b.silver]
        return next((c for c in priority if c in valid_cards), None)

    def discard_decision(self, prompt, card, valid_cards, player, game,
                         min_num_discard=0, max_num_discard=-1):
        remaining = list(enumerate(valid_cards))
        keep_count = len(remaining) - min_num_discard
        keep = set()
        # The next turn supplies one Action, so retain exactly one preferred terminal.
        if self.target and keep_count > 0:
            target_index = next((i for i, c in remaining if c == self.target), None)
            if target_index is not None:
                keep.add(target_index)
        def key(item):
            i, c = item
            money = c.money if CardType.Treasure in c.type else 0
            points = c.score(player) if hasattr(c, 'score') else 0
            return (-money, -points, i)
        for i, c in sorted(remaining, key=key):
            if len(keep) >= keep_count:
                break
            keep.add(i)
        return [c for i, c in remaining if i not in keep]


class StrongestBuiltInPolicy(BigMoneyUltimateDecider):
    """Expose pyminion's strongest general-purpose built-in policy as BMU-1."""

class Bridge:
    def __init__(self, session):
        self.session = session

    def ask(self, kind, values, player, game, minimum=0, maximum=1,
            source=None, instruction='', relevant=None, ordered=False):
        values = list(values)
        maximum = len(values) if maximum < 0 else min(maximum, len(values))
        minimum = min(minimum, maximum)  # impossible gains resolve as far as possible
        if maximum == 0:
            return []
        # Mandatory identical choices carry no strategic decision.
        if minimum == maximum == len(values) and (not ordered or len(values) == 1):
            return values[:]
        self.session.decision_id += 1
        prompt = {
            'kind': kind, 'source': source.name if source else None,
            'instruction': instruction, 'min': minimum, 'max': maximum,
            'ordered': ordered, 'duplicates': False,
            'options': [{'id': str(i), 'label': v.name if hasattr(v, 'name') else str(v)}
                        for i, v in enumerate(values)],
            'relevant_cards': [c.name for c in relevant or []],
        }
        self.session.publish(prompt)
        line = sys.stdin.readline()
        if not line:
            raise EOFError('controller closed')
        reply = json.loads(line)
        ids = reply['option_ids']
        assert reply['decision_id'] == self.session.decision_id
        assert minimum <= len(ids) <= maximum and len(ids) == len(set(ids))
        assert all(isinstance(i, str) and i.isdigit() and str(int(i)) == i
                   and 0 <= int(i) < len(values) for i in ids)
        return [values[int(i)] for i in ids]

    def action_phase_decision(self, valid_actions, player, game):
        result = self.ask('play_action', sorted(valid_actions, key=lambda c: c.name), player, game,
                          instruction='Play one Action, or pass to Treasures and buying.')
        return result[0] if result else None

    def treasure_phase_decision(self, valid_treasures, player, game):
        # Only ordinary Copper/Silver/Gold exist in this whitelist. No order-dependent choices.
        return list(valid_treasures)

    def buy_phase_decision(self, valid_cards, player, game):
        result = self.ask('buy', sorted(valid_cards, key=lambda c: c.name), player, game,
                          instruction='Buy one affordable card to discard, or pass to end buying.')
        return result[0] if result else None

    def binary_decision(self, prompt, card, player, game, relevant_cards=None):
        return self.ask('yes_no', [False, True], player, game, 1, 1,
                        card, prompt, relevant_cards)[0]

    def effects_order_decision(self, effects, player, game):
        names = [e.get_name() for e in effects]
        selected = self.ask('effect_order', names, player, game, 1, 1,
                            instruction='Choose the next triggered effect to resolve.')
        return names.index(selected[0])

    def selection(self, kind, prompt, card, valid_cards, player, game, minimum, maximum):
        # Preserve Sentry's inspected order; all other selections are unordered card multisets.
        cards = list(valid_cards) if card == b.sentry else sorted(valid_cards, key=lambda c: c.name)
        return self.ask(kind, cards, player, game, minimum, maximum, card, prompt)

    def trash_decision(self, prompt, card, valid_cards, player, game, min_num_trash=0, max_num_trash=-1):
        return self.selection('trash', prompt, card, valid_cards, player, game, min_num_trash, max_num_trash)

    def discard_decision(self, prompt, card, valid_cards, player, game, min_num_discard=0, max_num_discard=-1):
        return self.selection('discard', prompt, card, valid_cards, player, game, min_num_discard, max_num_discard)

    def gain_decision(self, prompt, card, valid_cards, player, game, min_num_gain=0, max_num_gain=-1):
        return self.selection('gain', prompt, card, valid_cards, player, game, min_num_gain, max_num_gain)


class Session:
    def __init__(self, config):
        self.config = config
        self.events = []
        self.decision_id = 0
        self.agent = PublicPlayer(Bridge(self), 'agent', self.events)
        if config['opponent_id'] == 'BMU-1':
            opponent_policy = StrongestBuiltInPolicy()
        elif config['opponent_id'] in ('BM-1', 'SM-1', 'WM-1'):
            # Retained solely for direct engine conformance tests/comparisons.
            opponent_policy = MoneyPolicy(config['opponent_id'])
        else:
            raise ValueError('Unknown opponent policy')
        self.opponent = PublicPlayer(opponent_policy, 'opponent', self.events)
        players = [self.agent, self.opponent]
        if config['agent_seat']:
            players.reverse()
        kingdom = [CARDS[n] for n in KINGDOMS[config['kingdom_id']]]
        self.game = Game(players, [list(CARDS.values())], kingdom_cards=kingdom,
                         random_order=False, log_stdout=False)
        self.finished = False
        self.truncated = False

    def summary(self):
        a, o = vp(self.agent.owned), vp(self.opponent.owned)
        outcome, win = 'ongoing', 0.0
        reason = None
        if self.truncated:
            outcome, reason = 'truncated', 'safety_limit'
        elif self.finished:
            winners = self.game.get_winners()
            outcome = 'draw' if len(winners) == 2 else 'win' if self.agent in winners else 'loss'
            win = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}[outcome]
            reason = 'province_empty' if self.game.supply.pile_length('Province') == 0 else 'three_piles_empty'
            assert a == self.agent.get_victory_points() and o == self.opponent.get_victory_points()
        return dict(agent_vp=a, opponent_vp=o, vp_margin=a-o, outcome=outcome,
                    win_score=win, end_reason=reason, truncated=self.truncated,
                    turns_taken={'agent': self.agent.turns, 'opponent': self.opponent.turns})

    def observation(self, prompt):
        g, a, o = self.game, self.agent, self.opponent
        summary = self.summary()
        supply = {}
        for pile in g.supply.piles:
            c = CARDS[pile.name]
            text = ' '.join((type(c).__doc__ or '').split())
            if c.name in ('Copper', 'Silver', 'Gold'):
                text = f'+${c.money} when played.'
            if c.name in ('Estate', 'Duchy', 'Province', 'Curse'):
                text = f'{c.score(a)} VP.'
            supply[c.name] = dict(count=len(pile), cost=c.get_cost(a, g).money,
                                  types=[t.name for t in c.type], text=text)
        def public(p):
            return dict(hand_count=len(p.hand), deck_count=len(p.deck),
                        discard_count=len(p.discard_pile), in_play=counts(p.playmat),
                        owned=dict(sorted((n, v) for n, v in p.owned.items() if v)),
                        vp=vp(p.owned))
        own = public(a)
        own.update(hand=counts(a.hand), discard=counts(a.discard_pile))
        state = g.current_player.state
        result = dict(episode_id='episode', decision_id=self.decision_id,
                      viewer='agent', agent_seat=self.config['agent_seat'],
                      active_player=g.current_player.player_id,
                      decision_player='agent' if prompt else None,
                      phase=g.current_phase.name, turns_taken=summary['turns_taken'],
                      resources=dict(actions=state.actions, buys=state.buys, coins=state.money),
                      kingdom_id=self.config['kingdom_id'], kingdom=KINGDOMS[self.config['kingdom_id']],
                      opponent_id=self.config['opponent_id'], supply=supply,
                      self=own, opponent=public(o), trash=counts(g.trash),
                      public_log_delta=self.events[:], prompt=prompt,
                      derived=dict(vp_margin=summary['vp_margin'], provinces_left=len(g.supply.get_pile('Province')),
                                   empty_piles=sum(len(p) == 0 for p in g.supply.piles)),
                      terminal=summary if self.finished else None)
        self.events.clear()
        return result

    def publish(self, prompt):
        obs = self.observation(prompt)
        info = self.summary()
        info.update(self.config)
        info.update(engine_version='pyminion 0.5.0', engine_commit=ENGINE_COMMIT,
                    adaptor_version=ADAPTOR_VERSION, valid_trial=True)
        print(json.dumps(dict(observation=obs, info=info, done=self.finished), allow_nan=False), flush=True)

    def run(self):
        random.seed(self.config['shuffle_seed'])
        self.game.start()
        while not self.finished:
            for player in self.game.players:
                self.game.current_player = player
                self.game.play_turn(player)
                if self.game.is_over():
                    self.finished = True
                    break
                if max(p.turns for p in self.game.players) >= self.config['max_turns']:
                    self.finished = self.truncated = True
                    break
        self.publish(None)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    apply()
    try:
        Session(json.loads(sys.stdin.readline())).run()
    except EOFError:
        pass
    except Exception:
        print(json.dumps({'error': traceback.format_exc()}), flush=True)
        sys.exit(1)
