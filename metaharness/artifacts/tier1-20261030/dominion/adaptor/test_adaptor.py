"""Rule conformance, hidden-information boundary, process lifecycle, and policy tests."""
import copy
import io
import json
import random

import pytest
from pyminion.core import CardType
from pyminion.expansions import base as b
from pyminion.player import State
from pyminion.bots.bot import BotDecider

from engine_patches import apply
from game import make_game
from scenarios import KINGDOMS, SCENARIOS, SEEDS
from worker import Session, MoneyPolicy, Bridge, counts, vp

apply()


class Script(BotDecider):
    def __init__(self, trash=(), discard=(), gain=None, binary=True):
        self.trashes, self.discards, self.gained, self.binary = list(trash), list(discard), gain, binary
        self.gain_seen = None

    def trash_decision(self, **kw):
        return self.trashes[:]

    def discard_decision(self, **kw):
        return self.discards[:]

    def gain_decision(self, **kw):
        self.gain_seen = (list(kw['player'].hand.cards), list(kw['game'].trash.cards), kw['valid_cards'])
        return [self.gained] if self.gained else kw['valid_cards'][:1]

    def binary_decision(self, **kw):
        self.relevant = kw.get('relevant_cards')
        return self.binary


def session(kingdom='curse_race'):
    s = Session(dict(kingdom_id=kingdom, opponent_id='WM-1' if kingdom == 'curse_race' else 'BM-1',
                     agent_seat=0, shuffle_seed=8, max_turns=400))
    random.seed(8)
    s.game.start()
    return s


def zone(player, hand=(), deck=(), discard=(), play=()):
    player.hand.cards = list(hand)
    player.deck.cards = list(deck)
    player.discard_pile.cards = list(discard)
    player.playmat.cards = list(play)
    player.set_aside.cards = []
    player.owned.clear()
    player.owned.update(c.name for c in player.get_all_cards())
    player.state = State()


@pytest.mark.parametrize('kingdom', KINGDOMS)
def test_setup(kingdom):
    s = session(kingdom)
    assert {p.name for p in s.game.supply.kingdom_piles} == set(KINGDOMS[kingdom])
    for p in s.game.players:
        assert counts(p.get_all_cards()) == {'Copper': 7, 'Estate': 3}
        assert len(p.hand) == len(p.deck) == 5
    for pile in s.game.supply.piles:
        expected = {'Estate': 8, 'Duchy': 8, 'Province': 8, 'Gardens': 8,
                    'Copper': 46, 'Silver': 40, 'Gold': 30}.get(pile.name, 10)
        assert len(pile) == expected


@pytest.mark.parametrize('card,draw,actions,money,buys', [
    (b.village, 1, 2, 0, 0), (b.smithy, 3, 0, 0, 0),
    (b.market, 1, 1, 1, 1), (b.festival, 0, 2, 2, 1),
    (b.laboratory, 2, 1, 0, 0), (b.moat, 2, 0, 0, 0),
    (b.merchant, 1, 1, 0, 0),
])
def test_simple_actions(card, draw, actions, money, buys):
    s = session('chapel_engine')
    p = s.agent
    zone(p, [card], [b.copper] * 10)
    p.play(card, s.game)
    assert len(p.hand) == draw
    assert p.state.actions == actions and p.state.money == money and p.state.buys == 1 + buys
    assert p.playmat.cards == [card]


def test_merchant_first_silver_and_no_carryover():
    s = session('chapel_engine')
    p = s.agent
    zone(p, [b.merchant, b.merchant, b.silver, b.silver], [b.copper] * 5)
    p.play(b.merchant, s.game)
    p.play(b.merchant, s.game)
    p.play(b.silver, s.game)
    assert p.state.money == 4
    p.play(b.silver, s.game)
    assert p.state.money == 6
    p.start_turn(s.game)
    p.hand.add(b.silver)
    p.play(b.silver, s.game)
    assert p.state.money == 2


@pytest.mark.parametrize('n', [0, 4])
def test_chapel(n):
    s = session()
    p = s.agent
    zone(p, [b.chapel] + [b.copper] * 4)
    p.decider = Script(trash=[b.copper] * n)
    p.play(b.chapel, s.game)
    assert len(s.game.trash) == n and len(p.hand) == 4 - n


def test_cellar_discard_then_draw_and_shuffle_only_when_needed():
    s = session('base_intro')
    p = s.agent
    zone(p, [b.cellar, b.estate, b.copper], [b.gold, b.silver])
    p.decider = Script(discard=[b.estate, b.copper])
    shuffles = p.shuffles
    p.play(b.cellar, s.game)
    assert counts(p.hand) == {'Gold': 1, 'Silver': 1} and p.shuffles == shuffles
    p.draw()
    assert p.shuffles == shuffles + 1


@pytest.mark.parametrize('card,target,gain,destination', [
    (b.remodel, b.estate, b.smithy, 'discard_pile'),
    (b.mine, b.copper, b.silver, 'hand'),
])
def test_trash_before_gain(card, target, gain, destination):
    s = session('base_intro')
    p = s.agent
    zone(p, [card, target])
    d = p.decider = Script(trash=[target], gain=gain)
    p.play(card, s.game)
    hand, trash, valid = d.gain_seen
    assert target not in hand and target in trash and gain in valid
    assert gain in getattr(p, destination).cards
    assert p.state.buys == 1 and p.state.money == 0


def test_mine_decline():
    s = session('base_intro')
    zone(s.agent, [b.mine, b.copper])
    s.agent.decider = Script()
    s.agent.play(b.mine, s.game)
    assert s.agent.hand.cards == [b.copper] and not s.game.trash.cards


def test_workshop_gain_and_multibuy():
    s = session('gardens_rush')
    p = s.agent
    zone(p, [b.workshop])
    p.decider = Script(gain=b.gardens)
    p.play(b.workshop, s.game)
    assert p.discard_pile.cards == [b.gardens]
    assert p.state.money == 0 and p.state.buys == 1
    p.state.money, p.state.buys = 6, 2
    p.buy(b.silver, s.game)
    p.buy(b.silver, s.game)
    assert p.state.money == p.state.buys == 0


@pytest.mark.parametrize('swap', [False, True])
def test_sentry_order_and_owned_zones(swap):
    s = session()
    p = s.agent
    zone(p, [b.sentry], [b.gold, b.estate, b.copper])
    d = p.decider = Script(binary=swap)
    p.play(b.sentry, s.game)
    assert d.relevant == [b.estate, b.gold]
    assert p.deck.cards == ([b.estate, b.gold] if swap else [b.gold, b.estate])
    assert not p.set_aside.cards and p.get_all_cards_count() == 4


def test_sentry_trash_and_discard():
    s = session()
    p = s.agent
    zone(p, [b.sentry], [b.gold, b.estate, b.copper])
    p.decider = Script(trash=[b.estate], discard=[b.gold])
    p.play(b.sentry, s.game)
    assert s.game.trash.cards == [b.estate] and p.discard_pile.cards == [b.gold]
    assert counts(p.get_all_cards()) == {k: v for k, v in p.owned.items() if v}


def test_moat_remaining_duplicate_and_hand_only():
    s = session()
    p, o = s.agent, s.opponent
    zone(p, [b.moat, b.moat], [b.copper] * 10)
    zone(o, [b.witch, b.witch])
    p.decider = Script(binary=True)
    p.play(b.moat, s.game)  # one Moat remains: upstream erroneously removed both reactions
    o.play(b.witch, s.game)
    assert p.get_card_count(b.curse) == 0 and s.game.supply.pile_length('Curse') == 10
    p.discard(s.game, b.moat)
    o.state.actions = 1
    o.play(b.witch, s.game)
    assert p.get_card_count(b.curse) == 1 and s.game.supply.pile_length('Curse') == 9


def test_militia_and_curse_exhaustion():
    s = session()
    zone(s.agent, [b.militia, b.witch], [b.copper] * 3)
    zone(s.opponent, [b.gold, b.silver, b.copper, b.estate, b.estate])
    s.opponent.decider = MoneyPolicy('BM-1')
    s.agent.play(b.militia, s.game)
    assert counts(s.opponent.hand) == {'Copper': 1, 'Gold': 1, 'Silver': 1}
    s.game.supply.get_pile('Curse').cards.clear()
    s.agent.state.actions = 1
    s.agent.play(b.witch, s.game)
    assert s.opponent.get_card_count(b.curse) == 0


def test_gardens_curses_all_zones_and_ties():
    s = session('gardens_rush')
    zone(s.agent, [b.gardens, b.curse], [b.copper] * 12, [b.gardens], [b.estate] * 5)
    assert s.agent.get_all_cards_count() == 20
    assert s.agent.get_victory_points() == vp(s.agent.owned) == 8
    zone(s.opponent, [b.province, b.estate, b.estate])
    s.agent.turns, s.opponent.turns = 5, 6
    assert s.game.get_winners() == [s.agent]
    s.opponent.turns = 5
    assert len(s.game.get_winners()) == 2


@pytest.mark.parametrize('end', ['province', 'three'])
def test_end_through_gains_finishes_turn(end):
    s = session('gardens_rush')
    p = s.agent
    zone(p, [b.workshop])
    p.decider = Script(gain=b.gardens)
    if end == 'province':
        s.game.supply.get_pile('Province').cards = [b.province]
        p.gain(b.province, s.game)
    else:
        s.game.supply.get_pile('Curse').cards.clear()
        s.game.supply.get_pile('Estate').cards.clear()
        s.game.supply.get_pile('Gardens').cards = [b.gardens]
        p.play(b.workshop, s.game)
    assert s.game.is_over()
    p.state.money, p.state.buys = 3, 1
    p.buy(b.silver, s.game)  # supply ending does not forbid remaining turn effects/buys
    assert b.silver in p.discard_pile.cards


def test_privacy_permutations_and_hidden_hands():
    s = session()
    first = s.observation(None)
    s.agent.deck.cards.reverse()
    s.opponent.hand.cards, s.opponent.deck.cards = s.opponent.deck.cards, s.opponent.hand.cards
    s.opponent.deck.cards.reverse()
    second = s.observation(None)
    assert first == second
    assert 'seed' not in json.dumps(second)


@pytest.mark.parametrize('policy,target', [('BM-1', None), ('SM-1', b.smithy), ('WM-1', b.witch)])
def test_opponent_priorities(policy, target):
    s = session()
    p, g = s.opponent, s.game
    d = MoneyPolicy(policy)
    p.state.money = 6
    valid = [c for c in g.supply.available_cards() if c.base_cost.money <= 6]
    assert d.buy_phase_decision(valid, p, g) == (target or b.gold)
    if target:
        p.gain(target, g)
        assert d.buy_phase_decision(valid, p, g) == b.gold
    g.supply.get_pile('Province').cards = [b.province] * 4
    assert d.buy_phase_decision(valid, p, g) == b.duchy
    g.supply.get_pile('Province').cards = [b.province] * 2
    assert d.buy_phase_decision([b.estate, b.silver], p, g) == b.estate
    assert d.buy_phase_decision([b.copper, b.curse], p, g) is None


def test_live_invalid_pause_replay_and_close():
    g, h = make_game(), make_game()
    try:
        obs = g.reset(1301)
        assert h.reset(1301) == obs
        pid = g._process.pid
        assert g.step('wait') == obs == g.observe()
        for bad in [None, {}, {'decision_id': 1, 'option_ids': [[]]}, 'choose 999 0', 'choose', 'bad']:
            current = g.observe()
            after = g.step(bad)
            for key in ('last_error', 'illegal_action_count'):
                after.pop(key)
                current.pop(key)
            assert after == current
        from evaluate import money_action
        for _ in range(50):
            if g.done:
                break
            action = money_action(g.observe())
            a, c = g.step(action), h.step(action)
            for key in ('last_error', 'illegal_action_count'):
                a.pop(key)
                c.pop(key)
            assert a == c
        process = g._process
        g.reset(1100)
        assert process.poll() is not None and g._process.pid != pid
    finally:
        process = g._process
        g.close()
        g.close()
        h.close()
    assert process.poll() is not None


def test_failure_and_truncation():
    g = make_game(max_turns=1)
    try:
        g.reset(1100)
        g.step('pass')
        assert g.done and g.info()['truncated'] and g.info()['outcome'] == 'truncated'
        assert g.info()['win_score'] == 0
    finally:
        g.close()
    g = make_game()
    try:
        g.reset(1100)
        g._process.kill()
        g._process.wait()
        g.step('pass')
        assert g.done and g.info()['valid_trial'] is False
    finally:
        g.close()


def test_seed_manifest():
    seeds = [seed for group in SEEDS.values() for seed in group]
    assert len(seeds) == len(set(seeds)) == len(SCENARIOS)
    for group in SEEDS.values():
        for seed in group:
            assert seed ^ 1 in group
    train = {SCENARIOS[s]['kingdom_id'] for s in SEEDS['train']}
    heldout = {SCENARIOS[s]['kingdom_id'] for s in SEEDS['heldout']}
    assert train.isdisjoint(heldout)


def test_no_eligible_gain_still_trashes():
    for card in (b.remodel, b.mine, b.workshop):
        s = session('base_intro')
        zone(s.agent, [card, b.copper])
        s.agent.decider = Script(trash=[b.copper])
        for pile in s.game.supply.piles:
            if pile.cards[0].base_cost.money <= 4:
                pile.cards.clear()
        s.agent.play(card, s.game)
        if card != b.workshop:
            assert s.game.trash.cards == [b.copper]
        assert not s.agent.discard_pile.cards


@pytest.mark.parametrize('attack', [b.witch, b.militia])
def test_bridge_choice_during_opponent_turn(attack, monkeypatch):
    s = session()
    p, o = s.agent, s.opponent
    zone(o, [attack], [b.copper] * 10)
    if attack == b.witch:
        zone(p, [b.moat, b.estate, b.copper, b.copper, b.silver])
        ids = ['1']  # True: reveal Moat
    else:
        zone(p, [b.gold, b.silver, b.copper, b.estate, b.estate])
        ids = ['1', '2']  # sorted prompt's two Estates
    s.game.current_player = o
    observations = []
    monkeypatch.setattr(s, 'publish', lambda prompt: observations.append(s.observation(prompt)))
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps({'decision_id': 1, 'option_ids': ids}) + '\n'))
    o.play(attack, s.game)
    assert len(observations) == 1
    obs = observations[0]
    assert obs['active_player'] == 'opponent' and obs['decision_player'] == 'agent'
    assert obs['prompt']['source'] == ('Moat' if attack == b.witch else 'Militia')
    if attack == b.witch:
        assert p.get_card_count(b.curse) == 0
    else:
        assert counts(p.hand) == {'Copper': 1, 'Gold': 1, 'Silver': 1}


def test_full_replay():
    from evaluate import run
    from replay import replay
    original = run(3100)
    reproduced = replay(original['replay_path'])
    for key in ('agent_vp', 'opponent_vp', 'vp_margin', 'outcome', 'win_score', 'turns_taken', 'end_reason'):
        assert reproduced[key] == original[key]
