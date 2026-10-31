"""Narrow Base-rule repairs to pyminion 0.5.0, applied in the worker only.

See ADAPTOR.md and regression tests. No files are modified at runtime.
"""
from pyminion.core import AbstractDeck, Action, CardType
from pyminion.effects import AttackEffect, EffectAction
from pyminion.expansions import base as b


def remodel(self, player, game, generic_play=True):
    Action.play(self, player, game, generic_play)
    if not player.hand.cards:
        return
    if len(player.hand) == 1:
        target = player.hand.cards[0]
    else:
        cards = player.decider.trash_decision(
            prompt='Trash one card from your hand.', card=self,
            valid_cards=player.hand.cards, player=player, game=game,
            min_num_trash=1, max_num_trash=1)
        target = cards[0]
    cost = target.get_cost(player, game) + 2
    player.trash(target, game)
    valid = [c for c in game.supply.available_cards() if c.get_cost(player, game) <= cost]
    if valid:
        gain = player.decider.gain_decision(
            prompt=f'Gain a card costing up to {cost} to discard.', card=self,
            valid_cards=valid, player=player, game=game, min_num_gain=1, max_num_gain=1)
        player.gain(gain[0], game)


def mine(self, player, game, generic_play=True):
    Action.play(self, player, game, generic_play)
    valid = [c for c in player.hand if CardType.Treasure in c.type]
    if not valid:
        return
    cards = player.decider.trash_decision(
        prompt='You may trash one Treasure from hand.', card=self,
        valid_cards=valid, player=player, game=game, min_num_trash=0, max_num_trash=1)
    if not cards:
        return
    target = cards[0]
    cost = target.get_cost(player, game) + 3
    player.trash(target, game)
    valid = [c for c in game.supply.available_cards()
             if CardType.Treasure in c.type and c.get_cost(player, game) <= cost]
    if valid:
        gain = player.decider.gain_decision(
            prompt=f'Gain a Treasure costing up to {cost} to hand.', card=self,
            valid_cards=valid, player=player, game=game, min_num_gain=1, max_num_gain=1)
        player.gain(gain[0], game, destination=player.hand)


def sentry(self, player, game, generic_play=True):
    Action.play(self, player, game, generic_play)
    # Keep temporary inspected cards in an owned zone for accurate intermediate VP.
    looked = player.set_aside
    assert not looked.cards
    player.draw(2, destination=looked, silent=True)  # list is first-to-draw order
    if looked.cards:
        trash = player.decider.trash_decision(
            prompt='Trash any of these inspected cards (listed top first).', card=self,
            valid_cards=looked.cards, player=player, game=game, min_num_trash=0, max_num_trash=2)
        for c in trash:
            player.trash(c, game, looked)
    if looked.cards:
        discard = player.decider.discard_decision(
            prompt='Discard any remaining inspected cards (listed top first).', card=self,
            valid_cards=looked.cards, player=player, game=game)
        for c in discard:
            player.discard(game, c, looked)
    reverse = False
    if len(looked) == 2:
        reverse = player.decider.binary_decision(
            prompt='Swap the two inspected cards? Relevant cards are currently top first.',
            card=self, player=player, game=game, relevant_cards=looked.cards[:])
    cards = looked.cards[:]
    if not reverse:
        cards.reverse()
    player.topdeck(cards, looked)


class MoatBlock(AttackEffect):
    def __init__(self):
        super().__init__('Moat: optional block', EffectAction.Other)

    def is_triggered(self, attacking_player, defending_player, attack_card, game):
        return b.moat in defending_player.hand.cards

    def handler(self, attacking_player, defending_player, attack_card, game):
        block = defending_player.decider.binary_decision(
            prompt=f'Reveal Moat to block {attack_card.name}?', card=b.moat,
            player=defending_player, game=game, relevant_cards=[attack_card])
        if block:
            defending_player.reveal(b.moat, game)
        return not block


def moat_setup(self, game):
    # One dynamic reaction per attack; duplicate Moats confer no extra protection.
    game.effect_registry.register_attack_effect(MoatBlock())


def workshop(self, player, game, generic_play=True):
    Action.play(self, player, game, generic_play)
    valid = [c for c in game.supply.available_cards() if c.get_cost(player, game) <= 4]
    if valid:
        cards = player.decider.gain_decision(
            prompt='Gain a card costing up to $4 to discard.', card=self,
            valid_cards=valid, player=player, game=game, min_num_gain=1, max_num_gain=1)
        player.gain(cards[0], game)


def apply():
    b.Remodel.play = remodel
    b.Mine.play = mine
    b.Sentry.play = sentry
    b.Moat.set_up = moat_setup
    b.Workshop.play = workshop
