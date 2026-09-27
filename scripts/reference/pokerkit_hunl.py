"""Reference heads-up no-limit hold'em trajectories from PokerKit (an independent
implementation, MIT, https://github.com/uoftcprg/pokerkit) for the differential tests
of the gmrules ``hunl`` engine (tests/test_hunl_reference.py).  Standalone: it runs
in its own throw-away uv environment, never in the project one:

    # PokerKit's own random hands
    uv run --no-project --python 3.12 --with pokerkit==0.7.6 \\
        python scripts/reference/pokerkit_hunl.py --hands 200 --seed 0 --out ref.json
    # gmrules trajectories replayed through PokerKit (JSON {"variants": [{"key",
    # "config", "trajectories": [[[seat, id], ...], ...]}]})
    uv run --no-project --python 3.12 --with pokerkit==0.7.6 \\
        python scripts/reference/pokerkit_hunl.py --replay ours.json --out ref.json

Every step records what PokerKit says, in the gmrules canonical ids (= OpenSpiel
``universal_poker`` full game): a chance step ``[-1, card]`` (card = rank * 4 + suit,
ranks 2..A, suits c d h s; hole cards seat 0, seat 0, seat 1, seat 1, then the
board), or an action step ``[seat, action, fold legal, check/call legal, raise min,
raise max]`` where the raise bounds are TOTAL commitments (PokerKit's street
"raise to" amount + what the player had committed on earlier streets; -1 = no
raise) and action 0 = fold, 1 = check / call, r >= 2 = raise to r; then the final
payoffs.  Seat 0 = PokerKit player 0 = the big blind (PokerKit reverses the blinds
heads-up, like the ACPC seats).  Mode TOURNAMENT: folding with nothing to call is
illegal (the ACPC rule); every PokerKit warning is an error.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import warnings
from typing import Dict, List, Optional

from pokerkit import Automation, Mode, NoLimitTexasHoldem

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FOLD, CALL = 0, 1

#: variant key → gmrules hunl config
VARIANTS: Dict[str, Dict[str, int]] = {
    "acpc": {"small_blind": 50, "big_blind": 100, "stack": 20000},
    "acpc2009": {"small_blind": 1, "big_blind": 2, "stack": 400},
    "short": {"small_blind": 1, "big_blind": 2, "stack": 13},
    "odd": {"small_blind": 3, "big_blind": 7, "stack": 150},
}

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def new_state(cfg: Dict[str, int]):
    return NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, (cfg["small_blind"], cfg["big_blind"]), cfg["big_blind"],
        (cfg["stack"], cfg["stack"]), 2, mode=Mode.TOURNAMENT)


def settle(state) -> None:
    """Burn cards (unknown) whenever PokerKit wants one: the deal is ours."""
    while state.status and state.can_burn_card():
        state.burn_card("??")


def committed_before_street(state, seat: int) -> int:
    return state.starting_stacks[seat] - state.stacks[seat] - state.bets[seat]


def kind_of(state) -> str:
    settle(state)
    if not state.status:
        return "terminal"
    if state.can_deal_hole() or state.can_deal_board():
        return "chance"
    if state.actor_index is not None:
        return "act"
    raise AssertionError("PokerKit state neither dealing, acting nor over")


def legal_of(state) -> List[int]:
    """[seat, fold, call, raise min, raise max] of the actor (total commitments)."""
    seat = state.actor_index
    base = committed_before_street(state, seat)
    lo = hi = -1
    if state.can_complete_bet_or_raise_to():
        lo = base + state.min_completion_betting_or_raising_to_amount
        hi = base + state.max_completion_betting_or_raising_to_amount
    return [seat, int(state.can_fold()), int(state.can_check_or_call()), lo, hi]


def apply_action(state, seat: int, action: int) -> None:
    assert seat == state.actor_index, (seat, state.actor_index)
    if action == FOLD:
        state.fold()
    elif action == CALL:
        state.check_or_call()
    else:
        state.complete_bet_or_raise_to(action - committed_before_street(state, seat))


def deal(state, n_dealt: int, card: int) -> None:
    """Deal the n-th card of the hand (hole cards: seat 0, seat 0, seat 1, seat 1)."""
    if n_dealt < 4:
        state.deal_hole(card_str(card), n_dealt // 2)
    else:
        state.deal_board(card_str(card))


def random_action(rng: random.Random, state, legal: List[int]) -> int:
    """fold 15% (when legal), raise 30% (when legal: min, all-in, pot fractions,
    uniform), else check / call."""
    seat, can_fold, _, lo, hi = legal
    if can_fold and rng.random() < 0.15:
        return FOLD
    if lo >= 0 and rng.random() < 0.3:
        k = rng.randrange(5)
        if k == 0:
            return lo
        if k == 1:
            return hi
        if k in (2, 3):
            m = max(committed_before_street(state, p) + state.bets[p] for p in range(2))
            pot = sum(state.starting_stacks[p] - state.stacks[p] for p in range(2))
            to_call = m - (committed_before_street(state, seat) + state.bets[seat])
            f = rng.choice([0.33, 0.5, 0.75, 1.0, 1.5, 2.0])
            return min(max(round(m + f * (pot + to_call)), lo), hi)
        return rng.randint(lo, hi)
    return CALL


def play(cfg: Dict[str, int], rng: Optional[random.Random] = None,
         events: Optional[List[List[int]]] = None) -> Dict:
    """One hand: PokerKit's own random play (``rng``) or the given events."""
    state = new_state(cfg)
    deck = list(range(52))
    if rng is not None:
        rng.shuffle(deck)
    steps: List[List[int]] = []
    i = n_dealt = 0
    while True:
        kind = kind_of(state)
        if kind == "terminal":
            break
        if events is not None and i >= len(events):
            return {"steps": steps, "final": kind, "returns": None}
        if kind == "chance":
            card = events[i][1] if events is not None else deck[n_dealt]
            if events is not None and events[i][0] != -1:
                return {"steps": steps, "final": kind, "returns": None, "error":
                        f"event {i}: PokerKit deals, got {events[i]}"}
            steps.append([-1, card])
            deal(state, n_dealt, card)
            n_dealt += 1
        else:
            legal = legal_of(state)
            if events is not None:
                seat, action = events[i]
            else:
                seat, action = legal[0], random_action(rng, state, legal)
            steps.append([legal[0], action] + legal[1:])
            if seat != legal[0]:
                return {"steps": steps, "final": kind, "returns": None, "error":
                        f"event {i}: seat {seat} acts, PokerKit says {legal[0]}"}
            try:
                apply_action(state, seat, action)
            except ValueError as e:
                return {"steps": steps, "final": kind, "returns": None,
                        "error": f"event {i}: {e}"}
        i += 1
    return {"steps": steps, "final": "terminal", "returns": [int(p) for p in state.payoffs]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", type=int, default=100, help="random hands per variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=sorted(VARIANTS))
    ap.add_argument("--replay", default=None, help="JSON of gmrules trajectories to replay")
    a = ap.parse_args()
    warnings.simplefilter("error")
    out = {"source": f"pokerkit {importlib.metadata.version('pokerkit')}", "variants": []}
    if a.replay:
        with open(a.replay) as fh:
            todo = [(v["key"], v["config"], v["trajectories"]) for v in json.load(fh)["variants"]]
        for key, cfg, trajs in todo:
            out["variants"].append({"key": key, "config": cfg,
                                    "trajectories": [play(cfg, events=t) for t in trajs]})
    else:
        for key in a.variants:
            rng = random.Random(f"{a.seed}:{key}")
            out["variants"].append({"key": key, "config": VARIANTS[key],
                                    "trajectories": [play(VARIANTS[key], rng=rng)
                                                     for _ in range(a.hands)]})
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
