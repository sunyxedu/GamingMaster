"""Frozen heuristic-v1: its only game input is a player's public/private view."""
from __future__ import annotations

import hashlib
import itertools
import math
import random

from pokerkit import Card, StandardHighHand

RANKS = "23456789TJQKA"
DECK = tuple(r + s for r in RANKS for s in "cdhs")
CARDS = {s: next(Card.parse(s)) for s in DECK}
PREMIUM = frozenset(("QQ", "KK", "AA", "AKs", "AKo"))
OPEN_RANGE = frozenset(
    [r + r for r in RANKS]
    + [hi + lo + "s" for i, hi in enumerate(RANKS) for lo in RANKS[:i]]
    + [hi + lo + "o" for hi in "AK" for lo in RANKS[:RANKS.index(hi)]]
    + "Q7o Q8o Q9o QTo QJo J8o J9o JTo T8o T9o 98o 87o".split()
)


def stream(seed: int, hand: int, domain: str) -> random.Random:
    digest = hashlib.sha256(f"poker-v1|{seed}|{hand}|{domain}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def hand_class(hole: list[str]) -> str:
    a, b = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    return a[0] + b[0] + ("" if a[0] == b[0] else "s" if a[1] == b[1] else "o")


def uniform_equity(hole: list[str], board: list[str], rng: random.Random) -> float:
    """512 independent samples, or all 990 opposing holes on the river.

    Actual opposing holes, burns and deck order are deliberately not inputs.
    PokerKit hand objects compare greater for stronger hands.
    """
    visible = set(hole + board)
    remaining = [CARDS[c] for c in DECK if c not in visible]
    own = tuple(CARDS[c] for c in hole)
    public = tuple(CARDS[c] for c in board)
    river = len(board) == 5
    samples = (itertools.combinations(remaining, 2) if river else
               (rng.sample(remaining, 7 - len(board)) for _ in range(512)))
    fixed = StandardHighHand.from_game(own, public) if river else None
    wins = 0.0
    n = 0
    for draw in samples:
        full_board = public + tuple(draw[2:])
        a = fixed if river else StandardHighHand.from_game(own, full_board)
        b = StandardHighHand.from_game(draw[:2], full_board)
        wins += 1.0 if a > b else 0.5 if a == b else 0.0
        n += 1
    return wins / n


def raise_target(view: dict, target: float) -> str:
    lo, hi = view["legal"]["min_raise_to"], view["legal"]["max_raise_to"]
    return f"raise_to {min(hi, max(lo, math.floor(target + 0.5)))}"


def pot_raise(view: dict, fraction: float) -> str:
    return raise_target(view, max(view["street_bets"]) +
                        fraction * (view["pot"] + view["to_call"]))


class HeuristicV1:
    version = "heuristic-v1"

    def __init__(self, mixing: random.Random, equity: random.Random):
        self._mixing = mixing
        self._equity_rng = equity
        self._cache: dict[tuple, float] = {}

    def choose(self, view: dict) -> str:
        key = tuple(view["hole_cards"] + view["board"])
        if key not in self._cache:
            self._cache[key] = uniform_equity(view["hole_cards"], view["board"], self._equity_rng)
        e = self._cache[key]
        legal = view["legal"]
        raising = legal["can_raise"]
        call = view["to_call"]
        q = call / (view["pot"] + call)
        history = [h for h in view["history"] if h["street"] == view["street"]]
        own = [h for h in history if h["seat"] == view["hero_seat"]]
        own_raises = sum(h["action"] == "raise_to" for h in own)
        if view["street"] == "preflop":
            cls = hand_class(view["hole_cards"])
            no_raise = not any(h["action"] == "raise_to" for h in history)
            if view["hero_seat"] == 1 and not own and no_raise:
                if raising:
                    return raise_target(view, 250) if cls in OPEN_RANGE else "fold"
                return "call" if e >= 0.50 else "fold"
            if view["hero_seat"] == 0 and no_raise and not call:
                return raise_target(view, 400) if raising and e >= 0.60 else "check"
            if cls in PREMIUM and raising:
                target = (3 * max(view["street_bets"]) if not own_raises
                          else legal["max_raise_to"])
                return raise_target(view, target)
            if not call:
                return "check"
            return "call" if e >= max(0.44, q + 0.08) else "fold"
        if not call:
            if raising and e >= 0.62:
                return pot_raise(view, 1 if e >= 0.85 else 0.75)
            if raising and self._mixing.random() < 0.08:
                return pot_raise(view, 0.5)
            return "check"
        if raising and e >= 0.80 and not own_raises:
            return pot_raise(view, 0.75)
        margin = 0 if view["street"] == "river" else 0.05
        return "call" if e >= q + margin else "fold"
