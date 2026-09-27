"""Causal hand-cycle belief — the Python twin of crdata's ``hand::CycleBelief``.

Clash Royale deals 4 of the 8 deck cards into the hand and queues the other 4; a
played card's slot is refilled from the queue front and the played card goes to
the queue back.  Of the 70 × 24 initial (hand set, queue order) states — equally
likely after the shuffle — the belief keeps those under which every own play SO
FAR was in hand, and reports per deck card the fraction of them in which it is in
hand (``in_hand``, sums to 4) and at the queue front (``next``, sums to 1).  Only
plays already made enter it, so it never knows more than the player (who sees the
true hand and next card); after the 4th in-deck play it is exact (0/1).

The converter writes it per frame as ``raw.deck_in_hand`` / ``raw.deck_next`` (the
own plays strictly before the frame's tick), over the deck tokens in
:func:`deck_token_order`.  DEPLOYMENT: the live bot must compute the network's
inputs the same way — one :class:`CycleBelief` per match, fed its own plays — and
not the client's true hand: the network was trained on this posterior, the true
hand is a different input distribution.  The client's true hand (and elixir) may
be used as a legality mask on the card head (``legal["card"]``): the true hand is
always inside the belief's support, so the mask only removes cards the belief
still allows.

Outputs are bit-identical to the converter's (same integer counts, one float64
division, then float32); ``tests/test_clash_royale_belief.py`` pins it on the real
replay fixture.
"""
from __future__ import annotations

from itertools import combinations, permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DECK = 8
HAND = 4

#: :meth:`CycleBelief.observe` results
CYCLED, NOT_IN_DECK, INCONSISTENT = "cycled", "not_in_deck", "inconsistent"


def deck_token_order(ids: Sequence[int], forms: Optional[Sequence[int]] = None) -> List[int]:
    """Deck indices in token order: sorted by (card id, form, index) — the order of
    the converter's ``raw.deck_*`` columns and of the card head's labels."""
    forms = list(forms) if forms is not None else [0] * len(ids)
    return sorted(range(len(ids)), key=lambda i: (int(ids[i]), int(forms[i]), i))


def _initial_states() -> Dict[Tuple[int, Tuple[int, ...]], int]:
    out: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    for hand in combinations(range(DECK), HAND):
        mask = sum(1 << i for i in hand)
        rest = [i for i in range(DECK) if not mask >> i & 1]
        for queue in permutations(rest):
            out[(mask, queue)] = 1
    return out


class CycleBelief:
    """The belief over one player's hand cycle.

    ``deck``: the 8 card ids in token order (probabilities are reported in that
    order).  ``observe(card_id)`` conditions on one own play."""

    def __init__(self, deck: Sequence[int]):
        if len(deck) != DECK:
            raise ValueError(f"a deck has {DECK} cards, got {len(deck)}")
        self.deck = [int(c) for c in deck]
        # distinct current states (hand bitmask, queue front-first) -> number of
        # initial states that lead to them
        self._states = _initial_states()
        self.plays = 0          # in-deck plays observed
        self.skipped = 0        # plays of cards that are not in the deck
        self._refresh()

    def observe(self, card: int) -> str:
        """Condition on one own play of ``card`` (in play order).  With repeated ids
        the lowest in-hand index holding it is played.  Returns ``"cycled"``,
        ``"not_in_deck"`` (the cycle is unchanged) or ``"inconsistent"`` (no
        surviving state had the card in hand; the belief is left as it was)."""
        idx = [i for i, c in enumerate(self.deck) if c == int(card)]
        if not idx:
            self.skipped += 1
            return NOT_IN_DECK
        advanced: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        for (mask, q), w in self._states.items():
            i = next((i for i in idx if mask >> i & 1), None)
            if i is None:
                continue
            key = ((mask & ~(1 << i)) | (1 << q[0]), (q[1], q[2], q[3], i))
            advanced[key] = advanced.get(key, 0) + w
        if not advanced:
            return INCONSISTENT
        self._states = advanced
        self.plays += 1
        self._refresh()
        return CYCLED

    def _refresh(self) -> None:
        total = sum(self._states.values())
        hand = [0] * DECK
        nxt = [0] * DECK
        for (mask, q), w in self._states.items():
            for i in range(DECK):
                if mask >> i & 1:
                    hand[i] += w
            nxt[q[0]] += w
        # one float64 division, then float32 — as the converter
        self._in_hand = np.array([h / total for h in hand], dtype=np.float64).astype(np.float32)
        self._next = np.array([n / total for n in nxt], dtype=np.float64).astype(np.float32)

    @property
    def in_hand(self) -> np.ndarray:
        """float32 [8]: P(deck card in hand | the plays observed so far)."""
        return self._in_hand.copy()

    @property
    def next(self) -> np.ndarray:
        """float32 [8]: P(deck card is the next card | the plays observed so far)."""
        return self._next.copy()

    @property
    def n_consistent(self) -> int:
        """Initial (hand set, queue order) states consistent with every play so far."""
        return sum(self._states.values())

    @property
    def determined(self) -> bool:
        """Whether the current hand and queue are known exactly."""
        return len(self._states) == 1
