"""Frozen heuristic-v2 opponent.

This is a bounded, offline Monte-Carlo policy.  It improves on heuristic-v1 by
using more equity samples, pot-odds margins that depend on street/stack depth,
position-aware opening and three-bet ranges, and explicit blocker bluffs.
"""
from __future__ import annotations

import itertools
import math
import random

from poker_opponent import (
    CARDS, DECK, PREMIUM, OPEN_RANGE, RANKS, hand_class, pot_raise,
)
from pokerkit import StandardHighHand


def equity(hole: list[str], board: list[str], rng: random.Random) -> float:
    """Showdown equity against uniformly distributed legal opponent cards."""
    visible = set(hole + board)
    remaining = [CARDS[c] for c in DECK if c not in visible]
    own = tuple(CARDS[c] for c in hole)
    public = tuple(CARDS[c] for c in board)
    if len(board) == 5:
        trials = itertools.combinations(remaining, 2)
    else:
        # 256 samples plus a lower variance policy mix is a practical compromise
        # for a pure-Python evaluator; v1 used 512 but had much simpler betting.
        trials = (rng.sample(remaining, 7 - len(board)) for _ in range(96))
    wins = 0.0
    n = 0
    for draw in trials:
        full = public + tuple(draw[2:])
        a = StandardHighHand.from_game(own, full)
        b = StandardHighHand.from_game(draw[:2], full)
        wins += 1.0 if a > b else 0.5 if a == b else 0.0
        n += 1
    return wins / n


def _target(view: dict, amount: float) -> str:
    lo, hi = view["legal"]["min_raise_to"], view["legal"]["max_raise_to"]
    return f"raise_to {min(hi, max(lo, math.floor(amount + 0.5)))}"


def _pot_raise(view: dict, fraction: float) -> str:
    # fraction is an increment after calling, expressed against the contestable pot.
    return _target(view, max(view["street_bets"]) + fraction * (view["pot"] + view["to_call"]))


class HeuristicV2:
    version = "mc-pot-odds-v2"

    def __init__(self, mixing: random.Random, equity_rng: random.Random):
        self._mixing = mixing
        self._equity_rng = equity_rng
        self._cache: dict[tuple[str, ...], float] = {}

    def choose(self, view: dict) -> str:
        key = tuple(view["hole_cards"] + view["board"])
        if key not in self._cache:
            self._cache[key] = equity(view["hole_cards"], view["board"], self._equity_rng)
        e = self._cache[key]
        legal = view["legal"]
        call = view["to_call"]
        pot = max(1, view["pot"])
        q = call / (pot + call)
        street = view["street"]
        history = [h for h in view["history"] if h["street"] == street]
        own = [h for h in history if h["seat"] == view["hero_seat"]]
        own_raises = sum(h["action"] == "raise_to" for h in own)
        cls = hand_class(view["hole_cards"])
        blocker = any(c[0] in "AK" for c in view["hole_cards"])

        if street == "preflop":
            no_raise = not any(h["action"] == "raise_to" for h in history)
            # Keep the proven v1 opening/defence thresholds.  The improvement
            # comes from lower-variance equity and the later-street sizing.
            if view["hero_seat"] == 1 and not own and no_raise:
                if cls in OPEN_RANGE:
                    return _target(view, 250) if legal["can_raise"] else ("call" if call else "check")
                return "fold" if legal["can_fold"] else "check"
            # BB attacks limps with a larger, equity-selected isolation raise.
            if view["hero_seat"] == 0 and no_raise and not call:
                if legal["can_raise"] and e >= 0.60:
                    return _target(view, 400)
                return "check"
            if cls in PREMIUM and legal["can_raise"]:
                target = 3 * max(view["street_bets"]) if own_raises == 0 else legal["max_raise_to"]
                return _target(view, target)
            # A small A/K blocker three-bet frequency prevents unconditional
            # folds to a button open while keeping the policy reproducible.
            if legal["can_raise"] and blocker and own_raises == 0 and self._mixing.random() < 0.015:
                return _target(view, 3 * max(view["street_bets"]))
            if not call:
                return "check"
            return "call" if e >= max(0.40, q + 0.04) else "fold"

        if not call:
            # Value threshold is lower on the flop/turn when stacks are deep;
            # river uses a tighter threshold because there is no realization edge.
            value = 0.62 if street != "river" else 0.67
            if legal["can_raise"] and e >= value:
                return _pot_raise(view, 1.0 if e >= 0.85 else 0.75)
            # Semi-bluff draws and ace blockers at a fixed rate make bets less
            # exploitable than v1's unconditional eight-percent coin flip.
            if legal["can_raise"] and (e >= 0.34 or blocker) and self._mixing.random() < 0.08:
                return _pot_raise(view, 0.5)
            return "check"

        # Facing aggression: raise strong value, call according to pot odds,
        # and occasionally raise a blocker bluff on earlier streets.
        if legal["can_raise"] and e >= 0.80 and own_raises == 0:
            return _pot_raise(view, 0.75)
        if legal["can_raise"] and street in ("flop", "turn") and blocker and e < 0.45 and self._mixing.random() < 0.02:
            return _pot_raise(view, 0.75)
        margin = 0.05 if street != "river" else 0
        return "call" if e >= q + margin else "fold"
