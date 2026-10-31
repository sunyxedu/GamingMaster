"""Public Meta-Harness entrypoint for the retained Dominion adaptor.

The implementation lives in :mod:`game_v1`; keeping that module intact makes
the revision easy to compare with the previous adaptor while exposing the
required ``game.py`` contract to the evaluator.
"""
from game_v1 import (
    Dominion as PreviousDominion,
    TIER,
    DETERMINISTIC,
    EVAL_KWARGS,
    SEEDS,
)


class Dominion(PreviousDominion):
    def describe(self):
        text = super().describe().replace('BM-1, SM-1 or WM-1 opponent',
                                         'BMU-1 (pyminion BigMoneyUltimate) opponent')
        start = text.index('BM-1 buys Province,')
        end = text.index('Safety limits', start)
        policy = '''BMU-1 is the engine's BigMoneyUltimate policy in every scenario. It plays Smithy.
Its first affordable nonempty buy priority is: Province if total Treasure money in its deck
exceeds $15; Duchy at <5 Provinces; Estate at <3; Gold; Duchy at <7;
Smithy if Smithies owned < Treasure cards owned / 11; Silver; otherwise stop.
It can own multiple Smithies. It uses upstream defaults for forced choices, including
discarding the first required cards in its hand; it does not use hidden opponent information.
'''
        return text[:start] + policy + text[end:]


def make_game(**kwargs):
    return Dominion(**kwargs)

__all__ = ["Dominion", "TIER", "DETERMINISTIC", "EVAL_KWARGS", "SEEDS", "make_game"]
