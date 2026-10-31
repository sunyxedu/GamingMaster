"""Rules conformance tests; fixture access to private state is test-only."""
import copy
import json
import random
import unittest
from collections import Counter

from game import COLORS, COPIES, RANKS, SEEDS, HanabiGame, _Card


class HanabiConformance(unittest.TestCase):
    def setUp(self):
        self.g = HanabiGame()
        self.g.reset(SEEDS["dev"][0])

    def fixture_card(self, color, rank):
        return self.g._new_card((color, rank))

    def assert_conservation(self, game):
        actual = Counter(game._deck)
        actual.update((c.color, c.rank) for hand in game.hands for c in hand)
        actual.update((c["color"], c["rank"]) for c in game.discard_pile)
        actual.update((color, rank) for color, height in game.fireworks.items()
                      for rank in range(1, height + 1))
        expected = Counter({(c, r): COPIES[r] for c in COLORS for r in RANKS})
        self.assertEqual(actual, expected)
        ids = [c.card_id for hand in game.hands for c in hand]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deck_and_dealing_all_profiles(self):
        for players, hand_size, deck_size in [(2, 5, 40), (3, 5, 35), (4, 4, 34), (5, 4, 30)]:
            g = HanabiGame(players)
            obs = g.reset(201)
            self.assertEqual([len(h) for h in obs["hands"]], [hand_size] * players)
            self.assertEqual(obs["deck_size"], deck_size)
            self.assertEqual((g.current_player, g.information_tokens, g.life_tokens, g.score), (0, 8, 3, 0))
            self.assert_conservation(g)
            for seat, hand in enumerate(g.hands):
                self.assertEqual([c.card_id for c in hand], list(range(seat, players * hand_size, players)))

    def test_seed_sequence_reproducibility_and_reset(self):
        g, other = self.g, HanabiGame()
        initial = g.reset(201)
        self.assertEqual(initial, other.reset(201))
        rng = random.Random(19)
        for _ in range(30):
            if g.done:
                break
            action = rng.choice(g.observe()["legal_actions"])
            self.assertEqual(g.step(action), other.step(action))
        self.assertEqual(g.reset(201), initial)
        self.assertEqual(g.info()["reward"], 0)
        self.assertEqual(g.observe()["history"], [])
        self.assertNotEqual(initial, g.reset(202))

    def test_own_hand_and_deck_noninterference(self):
        g = self.g
        before = g.observe()
        g.hands[0][0].color, g.hands[0][0].rank, g._deck[0] = (*g._deck[0], (g.hands[0][0].color, g.hands[0][0].rank))
        g._deck.reverse()
        self.assertEqual(g.observe(), before)
        self.assertTrue(all(c["color"] is None and c["rank"] is None for c in before["hands"][0]))
        self.assertTrue(all(c["color"] in COLORS for c in before["hands"][1]))
        self.assertNotIn('"seed"', json.dumps(before))
        self.assertNotIn('"deck"', json.dumps(before))
        self.assert_conservation(g)

    def test_color_clues_all_matches_negative_and_repeated(self):
        g = self.g
        g.hands[1] = [self.fixture_card("R", 1), self.fixture_card("B", 1), self.fixture_card("R", 3)]
        original_deck = list(g._deck)
        g.step("REVEAL_COLOR 1 R")
        self.assertEqual([c.possible_colors for c in g.hands[1]], [{"R"}, set(COLORS)-{"R"}, {"R"}])
        self.assertEqual([c.color_clued for c in g.hands[1]], [True, False, True])
        self.assertEqual(g.observe()["history"][-1]["touched_card_ids"], [g.hands[1][0].card_id, g.hands[1][2].card_id])
        self.assertEqual(g._deck, original_deck)
        # Act for seat 1, then give the same still-legal clue again.
        g.step({"type": "REVEAL_RANK", "target": 0, "rank": g.hands[0][0].rank})
        g.step("REVEAL_COLOR 1 R")
        self.assertEqual(g.information_tokens, 5)
        self.assertEqual(g.turn, 3)

    def test_rank_clue_and_new_card_knowledge(self):
        g = self.g
        g.hands[1] = [self.fixture_card("R", 2), self.fixture_card("G", 1), self.fixture_card("B", 2)]
        g.step("REVEAL_RANK 1 2")
        survivors = g.hands[1][1:]
        self.assertEqual([c.possible_ranks for c in g.hands[1]], [{2}, {1, 3, 4, 5}, {2}])
        self.assertEqual([c.rank_clued for c in g.hands[1]], [True, False, True])
        g.step("DISCARD 0")
        self.assertEqual(g.hands[1][:-1], survivors)
        replacement = g.hands[1][-1]
        self.assertEqual(replacement.possible_ranks, set(RANKS))
        self.assertEqual(replacement.possible_colors, set(COLORS))
        self.assertFalse(replacement.rank_clued or replacement.color_clued)
        event = g.observe()["history"][-1]
        self.assertEqual(event["drawn_card_id"], replacement.card_id)
        self.assertNotIn("drawn_color", event)
        self.assertNotIn("drawn_rank", event)

    def test_illegal_actions_no_state_mutation(self):
        g = self.g
        bad = [None, [], 6, "PLAY", "PLAY x", "PLAY 0 1", "__not_an_action__",
               {}, {"type": []}, {"type": "PLAY", "card_index": True},
               {"type": "PLAY", "card_index": -1}, {"type": "PLAY", "card_index": 5},
               {"type": "DISCARD", "card_index": 0},
               {"type": "REVEAL_COLOR", "target": 0, "color": "R"},
               {"type": "REVEAL_COLOR", "target": 1, "color": "Z"},
               {"type": "REVEAL_RANK", "target": 999, "rank": 1},
               {"type": "REVEAL_RANK", "target": 1, "rank": True}]
        for action in bad:
            before = copy.deepcopy(g.__dict__)
            obs = g.step(action)
            self.assertIsNotNone(obs["error"])
            after = copy.deepcopy(g.__dict__)
            for state in (before, after):
                state.pop("_last_error")
                state.pop("_last_reward")
            self.assertEqual(before, after)
        g.information_tokens = 0
        self.assertFalse(any(a["type"].startswith("REVEAL") for a in g.observe()["legal_actions"]))
        before = g.turn
        g.step({"type": "REVEAL_RANK", "target": 1, "rank": g.hands[1][0].rank})
        self.assertEqual(g.turn, before)
        g.hands[1] = [self.fixture_card("R", 1)]
        g.information_tokens = 1
        g.step("REVEAL_COLOR 1 B")
        self.assertEqual(g.turn, before)
        g.step("REVEAL_RANK 1 5")
        self.assertEqual(g.turn, before)

    def test_play_success_skipped_duplicate_and_bomb_score(self):
        g = self.g
        g.hands[0][0] = self.fixture_card("R", 1)
        g.step("PLAY 0")
        self.assertEqual((g.score, g.life_tokens, g.info()["reward"]), (1, 3, 1))
        for rank in (3, 1, 4):
            g.hands[g.current_player][0] = self.fixture_card("R", rank)
            self.assertIn({"type": "PLAY", "card_index": 0}, g.observe()["legal_actions"])
            g.step("PLAY 0")
        self.assertEqual(len(g.discard_pile), 3)
        self.assertTrue(g.done)
        self.assertEqual((g.score, g.fireworks_sum, g.info()["reward"]), (0, 1, -1))
        self.assertEqual(g.observe()["end_reason"], "lives_exhausted")
        self.assertEqual(g.observe()["legal_actions"], [])
        # Immediate bomb-out omits a useless replacement.
        self.assertEqual(len(g.hands[g.current_player]), 4)
        before = copy.deepcopy(g.__dict__)
        g.step("PLAY 0")
        g.step(None)
        self.assertEqual(before, g.__dict__)

    def test_five_clue_recovery_cap_and_perfect(self):
        g = self.g
        for tokens in (7, 8):
            g.reset(201)
            g.fireworks["R"] = 4
            g.information_tokens = tokens
            g.hands[0][0] = self.fixture_card("R", 5)
            g.step("PLAY 0")
            self.assertEqual((g.fireworks["R"], g.information_tokens), (5, 8))
        g.fireworks = {c: 5 for c in COLORS}
        g.fireworks["B"] = 4
        g.hands[g.current_player][0] = self.fixture_card("B", 5)
        deck_before = len(g._deck)
        g.step("PLAY 0")
        self.assertEqual((g.done, g.score, g.observe()["end_reason"]), (True, 25, "perfect"))
        self.assertEqual(len(g._deck), deck_before)

    def test_final_round_every_profile_and_action_kind(self):
        for players in (2, 3, 4, 5):
            for kind in ("PLAY", "DISCARD", "REVEAL_RANK", "REVEAL_COLOR"):
                g = HanabiGame(players)
                g.reset(201)
                g._deck = [("R", 1)]
                g.information_tokens = 6
                g.step("DISCARD 0")
                self.assertEqual(g.final_turns_remaining, players)
                self.assertFalse(g.done)
                for i in range(players):
                    actor = g.current_player
                    self.assertEqual(actor, (i + 1) % players)
                    if kind == "PLAY":
                        # Fixture safe play so countdown is the only terminator.
                        color = COLORS[i]
                        g.hands[actor][0] = g._new_card((color, 1))
                        action = "PLAY 0"
                    elif kind == "DISCARD":
                        g.information_tokens = 6
                        action = "DISCARD 0"
                    else:
                        target = (actor + 1) % players
                        card = g.hands[target][0]
                        action = {"type": kind, "target": target,
                                  "rank" if kind == "REVEAL_RANK" else "color":
                                  card.rank if kind == "REVEAL_RANK" else card.color}
                    if i == 0:
                        g.step("bad")
                        self.assertEqual(g.final_turns_remaining, players)
                    g.step(action)
                    self.assertEqual(g.final_turns_remaining, players - i - 1)
                    self.assertEqual(g.done, i == players - 1)
                self.assertEqual(g.observe()["end_reason"], "deck_exhausted")
                self.assertEqual(g.score, g.fireworks_sum)

    def test_observation_is_detached_json(self):
        g = self.g
        action = next(a for a in g.observe()["legal_actions"] if a["type"] == "REVEAL_RANK")
        g.step(action)
        g.step("DISCARD 0")
        before = g.observe()
        mutated = g.observe()
        mutated["history"][0]["touched_card_ids"].append(-100)
        mutated["discard_pile"][0]["rank"] = 99
        mutated["hands"][0][0]["possible_colors"].clear()
        mutated["fireworks"]["R"] = 100
        self.assertEqual(before, g.observe())
        json.dumps(before, allow_nan=False)

    def test_noop_and_string_dictionary_equivalence(self):
        g = self.g
        before = g.observe()
        self.assertEqual(g.step(g.action_spec.noop), before)
        other = HanabiGame()
        other.reset(201)
        self.assertEqual(g.step("PLAY 0"), other.step({"type": "PLAY", "card_index": 0}))

    def test_full_random_and_deck_exhaustion_episodes(self):
        for players in (2, 3, 4, 5):
            for mode in ("random", "discard"):
                for repeat in range(3):
                    g = HanabiGame(players)
                    obs = g.reset(201)
                    rng = random.Random(repeat)
                    while not g.done:
                        self.assertLess(g.turn, 150)
                        self.assertLess(len(json.dumps(obs, allow_nan=False).encode()), 200000)
                        self.assert_conservation(g)
                        actions = obs["legal_actions"]
                        if mode == "random":
                            action = rng.choice(actions)
                        else:
                            choices = [a for a in actions if a["type"] == "DISCARD"]
                            action = rng.choice(choices or [a for a in actions if a["type"].startswith("REVEAL")])
                        old_score = g.score
                        obs = g.step(action)
                        self.assertIsNone(obs["error"])
                        self.assertEqual(g.info()["reward"], g.score - old_score)
                        self.assertEqual(obs, g.observe())
                        self.assertTrue(all(c["color"] is None and c["rank"] is None for c in obs["hands"][obs["observer"]]))
                    self.assert_conservation(g)
                    self.assertGreaterEqual(g.score, 0)
                    self.assertLessEqual(g.score, 25)
                    if mode == "discard":
                        self.assertEqual(obs["end_reason"], "deck_exhausted")
                        self.assertEqual(obs["deck_size"], 0)


if __name__ == "__main__":
    unittest.main()
