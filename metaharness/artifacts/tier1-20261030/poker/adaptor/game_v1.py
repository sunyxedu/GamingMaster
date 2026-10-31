"""Offline, turn-based heads-up no-limit Hold'em on PokerKit 0.7.6."""
from __future__ import annotations

import copy
import importlib.metadata
import math
import re
import statistics
from typing import Any

from metaharness.games.base import ActionSpec
from pokerkit import Automation, Mode, NoLimitTexasHoldem
from poker_opponent import DECK, HeuristicV1, pot_raise, raise_target, stream

TIER = 1
DETERMINISTIC = True
SEEDS = {
    "train": [1103, 2207, 3301, 4409],
    "heldout": [51001, 62003, 73009, 84011],
    "dev": [901, 902],
    "validation": [15101, 26203, 37309, 48407],
}
EVAL_KWARGS = {"hands": 100}
AUTOMATIONS = (
    Automation.ANTE_POSTING, Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
)
STREETS = ("preflop", "flop", "turn", "river")


def new_state(stacks=(20000, 20000), blinds=(50, 100)):
    return NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, blinds, blinds[1], stacks, 2, mode=Mode.TOURNAMENT)


class PokerGame:
    name = "Heads-up no-limit Texas Hold'em — heuristic-v1"
    action_spec = ActionSpec(
        names=("fold", "check", "call", "raise_to", "wait"),
        descriptions={
            "fold": "Surrender the hand; legal only facing a wager.",
            "check": "Pass when to_call is zero.",
            "call": "Match the wager, capped at your remaining stack.",
            "raise_to": "raise_to N: bet/raise to integer N chips TOTAL on this street.",
            "wait": "Do nothing; the game has no clock and remains at your decision.",
        },
        noop="wait",
        notes="Use a candidate id verbatim, or 'raise_to N', or a typed action object. "
              "Only actions permitted by the current observation are legal. "
              "Malformed/illegal actions leave the poker state unchanged and report an error.",
    )

    def __init__(self, hands: int = 100):
        if type(hands) is not int or not 2 <= hands <= 1000 or hands % 2:
            raise ValueError("hands must be an even integer from 2 to 1000; primary benchmark = 100")
        if importlib.metadata.version("pokerkit") != "0.7.6":
            raise RuntimeError("This adaptor requires pokerkit==0.7.6")
        self.required_hands = hands
        self._ready = False
        self._closed = False
        self._failure = None
        self._payoffs: list[int] = []
        self._illegal = 0
        self._error = None
        self._last_result = None
        self._state = None

    def reset(self, seed: int) -> dict:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._seed = seed
        self._ready = True
        self._closed = False
        self._failure = None
        self._payoffs = []
        self._illegal = 0
        self._error = None
        self._last_result = None
        try:
            self._start_hand()
            self._advance()
        except Exception as exc:
            self._failure = f"engine failure: {type(exc).__name__}"
            raise
        return self.observe()

    def _start_hand(self):
        index = len(self._payoffs)
        self._hero = 1 - index % 2  # exactly balanced, first hand button
        self._deck = list(DECK)
        stream(self._seed, index, "deck").shuffle(self._deck)
        self._holes = [self._deck[:2], self._deck[2:4]]
        self._board: list[str] = []
        self._cursor = 4
        self._history: list[dict] = []
        self._state = new_state()
        for seat in range(2):
            self._state.deal_hole("".join(self._holes[seat]), seat)
        self._opponent = HeuristicV1(stream(self._seed, index, "opponent-mixing"),
                                     stream(self._seed, index, "opponent-equity"))

    def _advance(self):
        # Finite chip stacks bound betting, but guard engine deadlocks explicitly.
        for _ in range(200000):
            s = self._state
            if not s.status:
                self._settle()
                if self.done:
                    return
                self._start_hand()
                continue
            if s.can_burn_card():
                s.burn_card(self._deck[self._cursor])
                self._cursor += 1
            elif s.can_deal_board():
                card = self._deck[self._cursor]
                s.deal_board(card)
                self._board.append(card)
                self._cursor += 1
            elif s.actor_index is not None:
                if s.actor_index == self._hero:
                    return
                # The policy receives only this copied, filtered player view.
                view = self._view(s.actor_index, candidates=False)
                action = self._opponent.choose(view)
                self._apply(self._parse(action))
            else:
                raise RuntimeError("Active engine has no actor or pending deal")
        raise RuntimeError("Engine event bound exceeded")

    def _settle(self):
        s = self._state
        payoff = [int(x) for x in s.payoffs]
        if sum(payoff) != 0 or sum(s.stacks) != 40000:
            raise RuntimeError("Non-conserving engine settlement")
        # A folded or mucked hand is never revealed from our omniscient ledger.
        revealed = {str(i): list(self._holes[i]) for i in range(2)
                    if len(s.hole_card_statuses[i]) == 2 and all(s.hole_card_statuses[i])}
        self._last_result = {
            "hand_id": f"hand-{len(self._payoffs) + 1}", "hero_seat": self._hero,
            "hole_cards": list(self._holes[self._hero]), "board": list(self._board),
            "revealed_hole_cards": revealed, "net_chips": payoff[self._hero],
            "final_stacks": list(s.stacks), "history": copy.deepcopy(self._history),
        }
        self._payoffs.append(payoff[self._hero])

    def _view(self, seat: int, candidates=True) -> dict:
        s = self._state
        active = bool(s.status and s.actor_index is not None)
        committed = [20000 - x for x in s.stacks] if s.status else [0, 0]
        call = min(s.stacks[seat], max(s.bets) - s.bets[seat]) if active else 0
        raising = active and s.can_complete_bet_or_raise_to()
        legal = {
            "can_fold": active and s.can_fold(),
            "can_check": active and call == 0 and s.can_check_or_call(),
            "can_call": active and call > 0 and s.can_check_or_call(),
            "can_raise": bool(raising),
            "min_raise_to": s.min_completion_betting_or_raising_to_amount if raising else None,
            "max_raise_to": s.max_completion_betting_or_raising_to_amount if raising else None,
        }
        view = {
            "hero_seat": seat, "button_seat": 1, "actor_seat": s.actor_index,
            "street": STREETS[s.street_index] if s.status else "terminal",
            "hole_cards": list(self._holes[seat]), "board": list(self._board),
            "small_blind": 50, "big_blind": 100, "starting_stacks": [20000, 20000],
            "stacks": list(s.stacks), "street_bets": list(s.bets),
            "committed_total": committed, "pot": sum(committed), "to_call": call,
            "effective_stack": min(s.stacks), "legal": legal,
            "history": copy.deepcopy(self._history),
        }
        if candidates:
            choices = []
            for name in ("fold", "check", "call"):
                if legal[f"can_{name}"]:
                    choices.append({"id": name, "action": name,
                                    "cost": call if name == "call" else 0})
            if raising:
                ids = {raise_target(view, legal["min_raise_to"]),
                       raise_target(view, legal["max_raise_to"])}
                ids.update(pot_raise(view, f) for f in (1/3, 0.5, 0.75, 1, 1.5))
                if view["street"] == "preflop":
                    ids.update(raise_target(view, x) for x in (250, 300, 400))
                for identifier in sorted(ids, key=lambda x: int(x.split()[1])):
                    target = int(identifier.split()[1])
                    if s.can_complete_bet_or_raise_to(target):
                        choices.append({"id": identifier, "action": "raise_to", "amount": target,
                                        "cost": target - s.bets[seat]})
            view["candidates"] = choices
        return view

    @staticmethod
    def _parse(action: Any) -> tuple[str, int | None]:
        if isinstance(action, str):
            if len(action) > 64:
                raise ValueError("Action string too long")
            text = action.strip()
            if text in ("fold", "check", "call", "wait"):
                return text, None
            match = re.fullmatch(r"raise_to ([0-9]{1,9})", text)
            if match:
                return "raise_to", int(match[1])
        elif isinstance(action, dict):
            name = action.get("action")
            if isinstance(name, str):
                if name in ("fold", "check", "call", "wait") and set(action) == {"action"}:
                    return name, None
                if (name == "raise_to" and set(action) == {"action", "amount"}
                        and type(action["amount"]) is int):
                    return name, action["amount"]
        raise ValueError("Use fold/check/call/wait, 'raise_to N', or a typed action object")

    def _apply(self, parsed: tuple[str, int | None]):
        name, amount = parsed
        s = self._state
        seat = s.actor_index
        if seat is None:
            raise ValueError("No player is acting")
        call = min(s.stacks[seat], max(s.bets) - s.bets[seat])
        if name == "fold" and s.can_fold():
            operation, cost = s.fold, 0
        elif name in ("check", "call") and (name == "check") == (call == 0) and s.can_check_or_call():
            operation, cost = s.check_or_call, call
        elif name == "raise_to" and s.can_complete_bet_or_raise_to(amount):
            operation, cost = lambda: s.complete_bet_or_raise_to(amount), amount - s.bets[seat]
        else:
            raise ValueError("Action is not legal; use the current legal bounds or candidates")
        pot = sum(20000 - x for x in s.stacks)
        entry = {"seat": seat, "street": STREETS[s.street_index], "action": name,
                 "raise_to": amount, "cost": cost, "pot_before": pot,
                 "pot_after_wager": pot + cost}
        operation()
        self._history.append(entry)

    def step(self, action: Any) -> dict:
        if not self._ready or self._closed or self.done:
            return self.observe()
        self._error = None
        try:
            parsed = self._parse(action)
            if parsed[0] == "wait":
                return self.observe()
            self._apply(parsed)
        except (ValueError, TypeError, OverflowError):
            self._illegal += 1
            self._error = "Illegal or malformed action ignored; choose a legal candidate or raise_to integer."
            return self.observe()
        try:
            self._advance()
        except Exception as exc:
            self._failure = f"engine failure: {type(exc).__name__}"
            raise
        return self.observe()

    def observe(self) -> dict:
        if not self._ready:
            return {"schema_version": 1, "status": "not_started", "done": False}
        view = self._view(self._hero)
        view.update({
            "schema_version": 1, "hand_id": f"hand-{min(len(self._payoffs) + 1, self.required_hands)}",
            "completed_hands": len(self._payoffs), "required_hands": self.required_hands,
            "hand_terminal": self.done, "done": self.done, "closed": self._closed,
            "score_status": "final" if self.done else "provisional_fixed_denominator",
            "score_chips_per_hand": self.score, "total_chips": sum(self._payoffs),
            "opponent": HeuristicV1.version, "illegal_actions": self._illegal,
            "action_error": self._error, "last_hand": copy.deepcopy(self._last_result),
            "public_stats": {"sample_hands": len(self._payoffs),
                             "wins": sum(p > 0 for p in self._payoffs),
                             "losses": sum(p < 0 for p in self._payoffs),
                             "ties": sum(p == 0 for p in self._payoffs)},
        })
        return view

    @property
    def done(self) -> bool:
        return len(self._payoffs) == self.required_hands

    @property
    def score(self) -> float:
        # A partial run must not earn the mean of a profitable completed prefix.
        return float(sum(self._payoffs) / self.required_hands)

    def info(self) -> dict:
        n = len(self._payoffs)
        return {
            "valid_trial": self._failure is None,
            "invalid_reason": self._failure,
            "complete": self.done, "completed_hands": n, "required_hands": self.required_hands,
            "score_eligible": self.done and self._failure is None,
            "score_status": "final" if self.done else "incomplete",
            "total_chips": sum(self._payoffs), "chips_per_hand": self.score,
            "bb_per_hand": self.score / 100, "bb_per_100": self.score,
            "illegal_actions": self._illegal, "opponent": HeuristicV1.version,
            "engine": "pokerkit-0.7.6", "deal_design": "independent_balanced_seats",
            "uncertainty": {
                "unit": "independent_match_seed", "match_blocks": int(self.done),
                "standard_error_chips_per_hand": None,
                "reason": "Aggregate completed matches across independent seeds; hero may adapt within a match.",
                "descriptive_hand_sd_chips": statistics.stdev(self._payoffs) if n > 1 else None,
            },
            "terminal_payoffs_chips": list(self._payoffs),
        }

    def score_breakdown(self) -> dict[str, float]:
        return {"chips_per_hand": self.score, "total_chips": float(sum(self._payoffs)),
                "bb_per_hand": self.score / 100, "bb_per_100": self.score,
                "completed_hands": float(len(self._payoffs)),
                "required_hands": float(self.required_hands), "complete": float(self.done),
                "illegal_actions": float(self._illegal)}

    def describe(self) -> str:
        return """Play heads-up no-limit Texas Hold'em against the frozen heuristic-v1 opponent.
No clock, rake, ante, elimination or blind escalation. Every hand starts with 20000 chips
per player (200 big blinds); blinds are 50/100. Engine seat 0 is BB, seat 1 button/SB.
Button acts first preflop; BB acts first after the flop. Best five of seven cards wins;
ties split the pot. Folding when a check is free is disallowed. PokerKit handles minimum
full raises, short all-ins, calls, refunds and settlement. Seats alternate every hand.
Deals are independent; neither player sees seeds, burns, future cards or unrevealed holes.

Objective: maximize terminal NET chips won / required_hands (100 in the primary benchmark).
Fresh stacks do not erase losses from the score. Before completion the displayed score
is provisional, using the full required-hand denominator; it is not a completed match score.
Each step applies your action and advances the opponent and dealing to your next turn,
possibly in a later hand. done is true only when all required hands have settled.

Actions: 'fold', 'check', 'call', 'raise_to N' (integer chips), or a candidate id verbatim.
Equivalent objects are {"action":"fold"}, {"action":"check"}, {"action":"call"},
{"action":"raise_to","amount":300}. N is your TOTAL wager on the CURRENT STREET,
including that street's previous bet, not an increment or total-hand commitment.
For example SB opening to 300 pays 250 more; the next full raise is at least to 500.
Every integer between legal min_raise_to and max_raise_to is available. A stack-sized
call uses 'call'; a legal all-in raise uses 'raise_to max_raise_to'. 'wait' is a no-op.
Invalid input changes no poker state, increments illegal_actions and sets action_error.

Observation: hole_cards are yours; board contains only exposed community cards. All arrays
of stacks, street_bets and committed_total use engine seat order [BB,SB]. pot includes
outstanding bets exactly once; to_call is capped at your stack; effective_stack is the
minimum remaining stack. legal gives booleans and nullable current-street raise bounds.
candidates give exact action ids, raise amounts and incremental costs. history is this
hand's public betting sequence; pot_after_wager is before any refund or payout. last_hand
contains only your cards, the public board, legally shown cards, bets and your net result.
At terminal, stacks are after payout, pot/committed_total are zero, and legal is empty
of permitted actions. Progress, provisional/final score, public result counts and error
diagnostics are public. Future observations contain no secret engine state.

Opponent is nonlearning heuristic-v1: wide button openings, premium reraises, uniform
visible-card showdown equity (512 samples, exact river), price-sensitive calls, value
bets and an 8% mixed bet branch. It receives only its own holes and public information.
Its fixed policy is intentionally exploitable, not an expert or equilibrium benchmark.
Hero adaptation is allowed. Uncertainty must be assessed across independent match seeds,
not by treating correlated hands as independent. No duplicate deals occur within a match.
"""

    def close(self):
        # In-process engine: no servers, files, sockets or worker threads to release.
        self._closed = True


def make_game(**kwargs) -> PokerGame:
    return PokerGame(**kwargs)
