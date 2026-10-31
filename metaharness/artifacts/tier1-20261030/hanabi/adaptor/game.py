"""Offline, standard-library Hanabi adaptor for the Meta-Harness.

The engine intentionally keeps the real cards private and serializes only the
current seat's view.  It implements the two-player standard profile by default
and supports the ordinary 2--5 player hand-size variants.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True
SEEDS = {
    "train": [11, 12, 13, 14],
    "heldout": [101, 102, 103, 104],
    "dev": [201],
    "validation": [301, 302, 303],
}
EVAL_KWARGS = {"num_players": 2}

COLORS = ("R", "Y", "G", "W", "B")
RANKS = (1, 2, 3, 4, 5)
COPIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
RULES_VERSION = "hanabi-standard-offline-1"


@dataclass
class _Card:
    card_id: int
    color: str
    rank: int
    possible_colors: set[str]
    possible_ranks: set[int]
    color_clued: bool = False
    rank_clued: bool = False


class HanabiGame:
    name = "Hanabi (offline standard)"

    def __init__(self, num_players: int = 2) -> None:
        if type(num_players) is not int or num_players not in (2, 3, 4, 5):
            raise ValueError("num_players must be 2, 3, 4, or 5")
        self.num_players = int(num_players)
        hand_size = 5 if self.num_players <= 3 else 4
        self.action_spec = ActionSpec(
            names=["PLAY", "DISCARD", "REVEAL_COLOR", "REVEAL_RANK", "NOOP"],
            descriptions={
                "PLAY": "PLAY i: play own current slot i (may misplay and lose a life).",
                "DISCARD": "DISCARD i: discard own slot i to restore a clue; illegal at eight clues.",
                "REVEAL_COLOR": "REVEAL_COLOR seat color: spend one clue and mark all matches in that other seat's hand.",
                "REVEAL_RANK": "REVEAL_RANK seat rank: spend one clue and mark all matching ranks (1-5).",
                "NOOP": "Do nothing; does not consume a turn. This transport no-op is not a game pass.",
            },
            noop="NOOP",
            notes="Targets are absolute seat numbers. Card indices are current hand slots. A clue must match at least one card and marks every matching card; untouched cards receive negative information. No pass action exists.",
        )
        self._hand_size = hand_size
        self._closed = False
        self._seed = 0
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self._seed = int(seed)
        rng = random.Random(self._seed)
        identities = [(c, r) for c in COLORS for r in RANKS for _ in range(COPIES[r])]
        rng.shuffle(identities)
        self._deck = list(identities)
        self._next_card_id = 0
        self.hands: list[list[_Card]] = [[] for _ in range(self.num_players)]
        # Deal round-robin, assigning opaque IDs as cards enter the game.
        for _round in range(self._hand_size):
            for seat in range(self.num_players):
                self.hands[seat].append(self._new_card(self._deck.pop()))
        self.fireworks = {c: 0 for c in COLORS}
        self.discard_pile: list[dict[str, Any]] = []
        self.information_tokens = 8
        self.life_tokens = 3
        self.current_player = 0
        self.turn = 0
        self.final_turns_remaining: int | None = None
        self._done = False
        self._end_reason: str | None = None
        self._history: list[dict[str, Any]] = []
        self._last_error: str | None = None
        self._last_reward = 0.0
        self._closed = False
        return self.observe()

    def _new_card(self, identity: tuple[str, int]) -> _Card:
        card = _Card(self._next_card_id, identity[0], identity[1], set(COLORS), set(RANKS))
        self._next_card_id += 1
        return card

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        # During play the visible progress is useful; on a fuse-out the
        # benchmark score is explicitly zero even if stacks were built.
        return float(0 if self.life_tokens <= 0 else self.fireworks_sum)

    @property
    def fireworks_sum(self) -> int:
        return sum(self.fireworks.values())

    def _terminal_check(self) -> None:
        if self.life_tokens <= 0:
            self._done, self._end_reason = True, "lives_exhausted"
        elif self.fireworks_sum == 25:
            self._done, self._end_reason = True, "perfect"
        elif self.final_turns_remaining == 0:
            self._done, self._end_reason = True, "deck_exhausted"

    def _card_view(self, card: _Card, owner: int, observer: int, index: int) -> dict[str, Any]:
        own = owner == observer
        return {
            "card_id": card.card_id,
            "index": index,
            "color": None if own else card.color,
            "rank": None if own else card.rank,
            "possible_colors": sorted(card.possible_colors),
            "possible_ranks": sorted(card.possible_ranks),
            "color_clued": card.color_clued,
            "rank_clued": card.rank_clued,
        }

    def _legal_actions(self) -> list[dict[str, Any]]:
        if self._done:
            return []
        actor = self.current_player
        actions: list[dict[str, Any]] = []
        for i in range(len(self.hands[actor])):
            actions.append({"type": "PLAY", "card_index": i})
        if self.information_tokens < 8:
            for i in range(len(self.hands[actor])):
                actions.append({"type": "DISCARD", "card_index": i})
        for target in range(self.num_players):
            if target == actor:
                continue
            cards = self.hands[target]
            for color in COLORS:
                if any(c.color == color for c in cards) and self.information_tokens > 0:
                    actions.append({"type": "REVEAL_COLOR", "target": target, "color": color})
            for rank in RANKS:
                if any(c.rank == rank for c in cards) and self.information_tokens > 0:
                    actions.append({"type": "REVEAL_RANK", "target": target, "rank": rank})
        return actions

    def _action_key(self, action: Any) -> tuple[Any, ...] | None:
        if isinstance(action, str):
            parts = action.split()
            if not parts:
                return None
            kind = parts[0].upper()
            try:
                if kind in ("PLAY", "DISCARD") and len(parts) == 2:
                    action = {"type": kind, "card_index": int(parts[1])}
                elif kind == "REVEAL_COLOR" and len(parts) == 3:
                    action = {"type": kind, "target": int(parts[1]), "color": parts[2]}
                elif kind == "REVEAL_RANK" and len(parts) == 3:
                    action = {"type": kind, "target": int(parts[1]), "rank": int(parts[2])}
                else:
                    return None
            except ValueError:
                return None
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            return None
        kind = action["type"].upper()
        if kind in ("PLAY", "DISCARD"):
            i = action.get("card_index")
            if isinstance(i, bool) or not isinstance(i, int):
                return None
            return kind, i
        if kind == "REVEAL_COLOR":
            target, color = action.get("target"), action.get("color")
            if isinstance(target, bool) or not isinstance(target, int) or not isinstance(color, str):
                return None
            return kind, target, color.upper()
        if kind == "REVEAL_RANK":
            target, rank = action.get("target"), action.get("rank")
            if isinstance(target, bool) or not isinstance(target, int) or isinstance(rank, bool) or not isinstance(rank, int):
                return None
            return kind, target, rank
        return None

    def _valid_action(self, key: tuple[Any, ...] | None) -> bool:
        if key is None or self._done:
            return False
        kind = key[0]
        actor = self.current_player
        if kind in ("PLAY", "DISCARD"):
            i = key[1]
            return 0 <= i < len(self.hands[actor]) and (kind == "PLAY" or self.information_tokens < 8)
        if kind == "REVEAL_COLOR":
            _, target, color = key
            return 0 <= target < self.num_players and target != actor and color in COLORS and self.information_tokens > 0 and any(c.color == color for c in self.hands[target])
        if kind == "REVEAL_RANK":
            _, target, rank = key
            return 0 <= target < self.num_players and target != actor and rank in RANKS and self.information_tokens > 0 and any(c.rank == rank for c in self.hands[target])
        return False

    def _draw_replacement(self, actor: int) -> int | None:
        if self._deck:
            card = self._new_card(self._deck.pop())
            self.hands[actor].append(card)
            if not self._deck and self.final_turns_remaining is None:
                self.final_turns_remaining = self.num_players
            return card.card_id
        return None

    def step(self, action: Any) -> dict[str, Any]:
        if self._done:
            return self.observe()
        self._last_error = None
        self._last_reward = 0.0
        if isinstance(action, str) and action.strip().upper() == "NOOP":
            return self.observe()
        key = self._action_key(action)
        if not self._valid_action(key):
            self._last_error = "illegal or malformed action; state unchanged"
            return self.observe()
        assert key is not None
        actor = self.current_player
        kind = key[0]
        event: dict[str, Any] = {"turn": self.turn, "actor": actor, "type": kind}
        old_score = self.score
        started_final_round = False
        if kind in ("PLAY", "DISCARD"):
            index = key[1]
            card = self.hands[actor].pop(index)
            event.update({"card_id": card.card_id, "card_index": index, "color": card.color, "rank": card.rank})
            if kind == "PLAY":
                if self.fireworks[card.color] + 1 == card.rank:
                    self.fireworks[card.color] = card.rank
                    event["outcome"] = "played"
                    if card.rank == 5:
                        self.information_tokens = min(8, self.information_tokens + 1)
                else:
                    self.life_tokens -= 1
                    self.discard_pile.append({"card_id": card.card_id, "color": card.color, "rank": card.rank, "outcome": "misplay"})
                    event["outcome"] = "misplay"
            else:
                self.information_tokens = min(8, self.information_tokens + 1)
                self.discard_pile.append({"card_id": card.card_id, "color": card.color, "rank": card.rank, "outcome": "discard"})
                event["outcome"] = "discarded"
            self._terminal_check()
            if not self._done:
                drawn_id = self._draw_replacement(actor)
                if drawn_id is not None:
                    event["drawn_card_id"] = drawn_id
                    started_final_round = self.final_turns_remaining == self.num_players and not self._deck
        elif kind == "REVEAL_COLOR":
            _, target, color = key
            self.information_tokens -= 1
            touched = []
            for card in self.hands[target]:
                if card.color == color:
                    card.possible_colors.intersection_update({color})
                    card.color_clued = True
                    touched.append(card.card_id)
                else:
                    card.possible_colors.discard(color)
            event.update({"target": target, "color": color, "touched_card_ids": touched})
        else:
            _, target, rank = key
            self.information_tokens -= 1
            touched = []
            for card in self.hands[target]:
                if card.rank == rank:
                    card.possible_ranks.intersection_update({rank})
                    card.rank_clued = True
                    touched.append(card.card_id)
                else:
                    card.possible_ranks.discard(rank)
            event.update({"target": target, "rank": rank, "touched_card_ids": touched})
        self._history.append(event)
        self.turn += 1
        if self.final_turns_remaining is not None and not started_final_round:
            # The turn which drew the final card starts the countdown but is
            # itself not counted among the N closing actions.
            self.final_turns_remaining -= 1
        if not self._done:
            self._terminal_check()
        if not self._done:
            self.current_player = (self.current_player + 1) % self.num_players
        # Keep reward available to callers that want a Gym-like diagnostic.
        self._last_reward = self.score - old_score
        return self.observe()

    def observe(self) -> dict[str, Any]:
        observer = self.current_player
        return {
            "rules_version": RULES_VERSION,
            "num_players": self.num_players,
            "observer": observer,
            "current_player": observer,
            "turn": self.turn,
            "fireworks": dict(self.fireworks),
            "information_tokens": self.information_tokens,
            "life_tokens": self.life_tokens,
            "deck_size": len(self._deck),
            "final_turns_remaining": self.final_turns_remaining,
            "hands": [[self._card_view(c, seat, observer, i) for i, c in enumerate(hand)] for seat, hand in enumerate(self.hands)],
            "discard_pile": copy.deepcopy(self.discard_pile),
            "history": copy.deepcopy(self._history),
            "legal_actions": self._legal_actions(),
            "fireworks_sum": self.fireworks_sum,
            "score": self.score,
            "done": self._done,
            "end_reason": self._end_reason,
            "error": self._last_error,
        }

    def describe(self) -> str:
        return f"""Cooperative standard Hanabi, {self.num_players} seats, all controlled by the same policy.
Seat 0 starts; turns advance modulo the number of players. Each color R,Y,G,W,B has
three 1s, two each of 2/3/4, and one 5 (50 cards). Hands contain five cards for 2-3
players, four for 4-5. The TEAM shares eight clue tokens and three lives.
Build each firework in order 1,2,3,4,5. Playing any own card is legal: only the next
required rank succeeds; a duplicate or skipped rank is discarded and loses a life.
A discard restores one clue and is illegal at eight clues. A successful 5 restores
one clue, capped at eight. After a play/discard, draw a replacement if possible.
Clues cost one token, name one color or rank, target another absolute seat, and
must touch at least one card. Every matching card is identified; untouched cards
exclude that attribute. Repeated truthful clues are legal and still cost a token.
The final draw starts exactly N further actions, ending with the drawer's turn.
No replacements are drawn once the deck is empty. Zero lives or 25 points ends
immediately, without a replacement draw. Maximize terminal score 0-25: zero on
life exhaustion, otherwise the sum of the five heights. There are no bonuses.

Choose a dictionary from legal_actions, or use these compact string commands:
PLAY i; DISCARD i; REVEAL_COLOR seat color; REVEAL_RANK seat rank.
Equivalent JSON: {{"type":"PLAY","card_index":0}},
{{"type":"DISCARD","card_index":0}},
{{"type":"REVEAL_COLOR","target":1,"color":"R"}},
{{"type":"REVEAL_RANK","target":1,"rank":1}}.
Indices are zero-based oldest-to-newest slots; removing a card shifts later slots
left and a replacement is appended. Ranks are 1-5; targets are absolute seats.
NOOP leaves the game unchanged, without passing. Illegal/malformed actions do
not lose a turn: error reports rejection. Use a finite retry budget externally.

Observation fields: rules_version identifies this engine profile; num_players,
observer/current_player and turn identify the current view and valid action count.
fireworks maps colors to heights; information_tokens, life_tokens and deck_size
are public counts. final_turns_remaining is null until the final draw, then N,
decrementing after each subsequent valid action. hands is indexed by absolute seat.
Each card has opaque card_id, current index, color/rank (always null for your own
hand), possible_colors/possible_ranks, and color_clued/rank_clued positive flags.
The Cartesian product of these masks encodes literal positive AND negative clues,
not remaining card availability or conventions. New cards have all candidates.
discard_pile contains public discards/misplays. history contains public actions,
clue touched_card_ids, revealed play/discard faces and drawn_card_id only; it
never gives a drawn face. legal_actions is complete and empty when done.
fireworks_sum measures progress, score is current benchmark score (zero after
bomb-out), done/end_reason indicate terminal state, and error is null or a
rejection diagnostic. At terminal the last actor remains the observer.

Maintain separate private sessions/scratchpads per seat and episode. Never feed
one seat's visible hands or private reasoning to another seat; only public events
and preagreed conventions can be shared. Reset clears engine episode state; the
caller must clear its own sessions. Seeds, deck order and engine internals are
not policy inputs. Shared private memory would make the run omniscient.
"""

    def score_breakdown(self) -> dict[str, float]:
        return {"fireworks_sum": float(self.fireworks_sum), "benchmark_score": float(self.score), "life_tokens": float(self.life_tokens), "information_tokens": float(self.information_tokens)}

    def info(self) -> dict[str, Any]:
        return {"valid_trial": True, "seed": self._seed, "num_players": self.num_players,
                "suite": f"standard-{self.num_players}p", "rules_version": RULES_VERSION,
                "end_reason": self._end_reason, "reward": self._last_reward}

    def close(self) -> None:
        self._closed = True


def make_game(**kwargs: Any) -> HanabiGame:
    return HanabiGame(**kwargs)
