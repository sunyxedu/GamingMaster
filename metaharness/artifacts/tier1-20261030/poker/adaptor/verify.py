"""Acceptance tests against installed PokerKit; run with the evaluator Python."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import time
import unittest
import warnings

from pokerkit import StandardHighHand
from game import PokerGame, SEEDS, make_game, new_state
from poker_opponent import DECK, HeuristicV1, OPEN_RANGE, PREMIUM, stream, uniform_equity

ROOT = Path(__file__).resolve().parents[3]


def deal_holes(state, holes=("AcAd", "KcKd")):
    for seat, cards in enumerate(holes):
        state.deal_hole(cards, seat)


def runout(state, board="2c3d4h8s9c"):
    cards = [board[i:i+2] for i in range(0, len(board), 2)]
    used = {repr(c) for hand in state.hole_cards for c in hand} | set(cards)
    burns = iter(c for c in DECK if c not in used)
    board_cards = iter(cards)
    actors = []
    while state.status:
        if state.can_burn_card():
            state.burn_card(next(burns))
        elif state.can_deal_board():
            state.deal_board(next(board_cards))
        else:
            actors.append((state.street_index, state.actor_index))
            state.check_or_call()
    return actors


class EngineTests(unittest.TestCase):
    def test_blinds_fold_and_order(self):
        s = new_state()
        self.assertEqual(s.bets, [100, 50])
        deal_holes(s)
        self.assertEqual(s.actor_index, 1)
        s.fold()
        self.assertEqual(s.payoffs, [50, -50])
        self.assertFalse(any(any(x) for x in s.hole_card_statuses))
        s = new_state()
        deal_holes(s)
        s.check_or_call()
        self.assertEqual(s.actor_index, 0)
        self.assertFalse(s.can_fold())
        s.check_or_call()
        self.assertEqual(sum(20000 - x for x in s.stacks), 200)
        actors = runout(s)
        self.assertEqual(actors, [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)])
        self.assertEqual(s.payoffs, [100, -100])

    def test_raises_and_street_units(self):
        s = new_state()
        deal_holes(s)
        s.complete_bet_or_raise_to(300)
        self.assertEqual(s.stacks, [19900, 19700])
        self.assertEqual(s.min_completion_betting_or_raising_to_amount, 500)
        self.assertFalse(s.can_complete_bet_or_raise_to(499))
        s.check_or_call()
        s.burn_card("2c")
        s.deal_board("3c4c5d")
        self.assertEqual(s.bets, [0, 0])
        self.assertEqual(s.min_completion_betting_or_raising_to_amount, 100)
        s.complete_bet_or_raise_to(200)
        self.assertEqual(s.bets, [200, 0])
        self.assertEqual(s.stacks, [19500, 19700])

    def test_short_allin_and_refund(self):
        s = new_state(stacks=(350, 1000))
        deal_holes(s)
        s.complete_bet_or_raise_to(300)
        self.assertEqual(s.min_completion_betting_or_raising_to_amount, 350)
        s.complete_bet_or_raise_to(350)
        self.assertFalse(s.can_complete_bet_or_raise_to())
        s.check_or_call()
        self.assertEqual(runout(s), [])
        self.assertEqual(s.payoffs, [350, -350])
        self.assertEqual(sum(s.stacks), 1350)
        s = new_state(stacks=(350, 1000))
        deal_holes(s)
        s.complete_bet_or_raise_to(1000)
        self.assertFalse(s.can_complete_bet_or_raise_to())
        s.check_or_call()
        runout(s)
        self.assertEqual(s.payoffs, [350, -350])
        self.assertEqual(s.stacks, [700, 650])  # uncalled 650 refunded

    def test_showdown_ranking_and_ties(self):
        cases = [
            (("2c3d", "4c5d"), "AhKhQhJhTh", [0, 0]),  # board plays
            (("Ac2d", "KhKd"), "3c4h5s9dTc", [20000, -20000]),  # wheel
            (("Ah2h", "Tc9c"), "3h5h7h8dJs", [20000, -20000]),  # flush > straight
            (("AcAd", "KhQh"), "AsKdKs2h3h", [20000, -20000]),  # full house
            (("AcQd", "AhJd"), "As7h5c3d2h", [20000, -20000]),  # kicker
        ]
        for holes, board, payoffs in cases:
            with self.subTest(holes=holes, board=board):
                s = new_state()
                deal_holes(s, holes)
                s.complete_bet_or_raise_to(20000)
                s.check_or_call()
                self.assertEqual(runout(s, board), [])
                self.assertEqual(s.payoffs, payoffs)
                self.assertEqual(sum(s.stacks), 40000)
        self.assertGreater(StandardHighHand.from_game("Ac2d", "3c4h5s9dTc"),
                           StandardHighHand.from_game("KhKd", "3c4h5s9dTc"))

    def test_all_240_reference_trajectories(self):
        path = ROOT / "scripts/reference/pokerkit_hunl.py"
        spec = importlib.util.spec_from_file_location("poker_reference", path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        fixture = json.loads((ROOT / "tests/fixtures/hunl/pokerkit_reference.json").read_text())
        count = 0
        for direction in ("pokerkit_hands", "gmrules_hands_in_pokerkit"):
            for variant in fixture[direction]:
                for trajectory in variant["trajectories"]:
                    actual = ref.play(variant["config"], events=[x[:2] for x in trajectory["steps"]])
                    self.assertEqual(actual, trajectory)
                    count += 1
        self.assertEqual(count, 240)


class AdaptorTests(unittest.TestCase):
    def test_actions_are_strict_and_nonmutating_on_error(self):
        g = make_game(hands=2)
        g.reset(SEEDS["dev"][0])
        before = g._view(g._hero)
        invalid = [None, [], 3, True, {}, {"action": []}, "__bad__", "check",
                   "raise_to", "raise_to -10", "raise_to 199", "raise_to 20001",
                   {"action": "raise_to", "amount": True},
                   {"action": "raise_to", "amount": 300.0},
                   {"action": "raise_to", "amount": "300"},
                   {"action": "raise_to", "amount": 10**1000},
                   {"action": "call", "extra": 1}]
        for action in invalid:
            view = g.step(action)
            self.assertIsNotNone(view["action_error"])
            self.assertEqual(before, g._view(g._hero))
            json.dumps(view, allow_nan=False)
        self.assertEqual(g.info()["illegal_actions"], len(invalid))
        self.assertTrue(g.info()["valid_trial"])
        self.assertFalse(g.info()["score_eligible"])
        g.step("wait")
        self.assertEqual(before, g._view(g._hero))
        for action, parsed in [("raise_to 300", ("raise_to", 300)),
                               ({"action": "raise_to", "amount": 300}, ("raise_to", 300)),
                               ({"action": "call"}, ("call", None))]:
            self.assertEqual(g._parse(action), parsed)
        g.close()
        g.close()

    def test_privacy_and_copying(self):
        g = make_game(hands=2)
        original = g.reset(SEEDS["dev"][0])
        original["hole_cards"][0] = "XX"
        self.assertNotIn("XX", g.observe()["hole_cards"])
        for seat in (0, 1):
            before = g._view(seat)
            saved = g._holes[1-seat]
            g._holes[1-seat] = ["XX", "YY"]
            g._deck[12:] = reversed(g._deck[12:])
            self.assertEqual(before, g._view(seat))
            g._holes[1-seat] = saved
        self.assertFalse({"seed", "deck", "burns", "rng", "opponent_holes"} & g.observe().keys())
        # Equity takes only visible inputs; a board royal flush is always a tie.
        self.assertEqual(uniform_equity(["2c", "3d"], list(("Ah", "Kh", "Qh", "Jh", "Th")),
                                        random.Random(1)), 0.5)
        # Consuming an arbitrary number of policy draws cannot change the deck stream.
        a, b = list(DECK), list(DECK)
        stream(901, 0, "deck").shuffle(a)
        r = stream(901, 0, "opponent-equity")
        for _ in range(10000):
            r.random()
        stream(901, 0, "deck").shuffle(b)
        self.assertEqual(a, b)

    def test_opponent_policy_branches(self):
        g = make_game(hands=2)
        g.reset(901)
        v = g._view(1)

        def choose(view, equity, mixing=1.0):
            class FixedMix:
                def random(self):
                    return mixing
            p = HeuristicV1(FixedMix(), random.Random(1))
            p._cache[tuple(view["hole_cards"] + view["board"])] = equity
            return p.choose(view)

        v["hole_cards"] = ["Ac", "Kd"]
        self.assertEqual(choose(v, 0.7), "raise_to 250")
        v["hole_cards"] = ["2c", "7d"]
        self.assertEqual(choose(v, 0.3), "fold")
        v.update(street="flop", board=["Ah", "Ks", "Qc"], to_call=0,
                 street_bets=[0, 0], pot=600)
        v["legal"].update(min_raise_to=100)
        self.assertEqual(choose(v, 0.62), "raise_to 450")
        self.assertEqual(choose(v, 0.85), "raise_to 600")
        self.assertEqual(choose(v, 0.2, 0.079), "raise_to 300")
        self.assertEqual(choose(v, 0.2, 0.08), "check")
        v.update(to_call=300, street_bets=[300, 0], pot=900)
        v["legal"].update(min_raise_to=600)
        self.assertEqual(choose(v, 0.8), "raise_to 1200")
        self.assertEqual(choose(v, 0.3), "call")
        self.assertEqual(choose(v, 0.29), "fold")
        self.assertEqual(PREMIUM, frozenset("QQ KK AA AKs AKo".split()))
        self.assertIn("32s", OPEN_RANGE)
        self.assertNotIn("72o", OPEN_RANGE)

    def test_full_match_determinism_and_score(self):
        games = [make_game(hands=4), make_game(hands=4)]
        observations = [g.reset(902) for g in games]
        self.assertEqual(*observations)
        steps = 0
        while not games[0].done:
            action = "call" if observations[0]["legal"]["can_call"] else "check"
            observations = [g.step(action) for g in games]
            self.assertEqual(*observations)
            self.assertLess(len(json.dumps(observations[0], allow_nan=False)), 200000)
            steps += 1
            self.assertLess(steps, 200)
        for g in games:
            info = g.info()
            self.assertEqual(g.score, sum(info["terminal_payoffs_chips"]) / 4)
            self.assertEqual(info["completed_hands"], 4)
            self.assertTrue(info["score_eligible"])
            self.assertIsNone(info["uncertainty"]["standard_error_chips_per_hand"])
            terminal = g.observe()
            self.assertEqual(g.step("call"), terminal)
            self.assertEqual(g.reset(902), make_game(hands=4).reset(902))
        g = make_game(hands=4)
        g.reset(901)
        # Apply and settle just one immediate SB fold without auto-starting the next hand.
        g._apply(("fold", None))
        g._settle()
        self.assertEqual(g._payoffs, [-50])
        self.assertEqual(g.score, -12.5)
        self.assertFalse(g.done)
        self.assertEqual(g._last_result["revealed_hole_cards"], {})

    def test_adaptor_allin_runout(self):
        g = make_game(hands=2)
        g.reset(901)
        # Testing double only: force the opponent to call to exercise adapter chance advancement.
        g._opponent.choose = lambda view: "call"
        g.step("raise_to 20000")
        self.assertGreaterEqual(len(g._payoffs), 1)
        self.assertIn(g._payoffs[0], [-20000, 0, 20000])
        self.assertEqual(g.info()["illegal_actions"], 0)


if __name__ == "__main__":
    warnings.simplefilter("error")
    unittest.main(verbosity=2)
