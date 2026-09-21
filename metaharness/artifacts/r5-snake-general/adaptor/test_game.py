"""Rule regressions, seeded local gameplay and the actual Meta-Harness wall clock."""
import json
import random
import time
import types
import unittest
from pathlib import Path

from game import DIRECTIONS, PROFILES, make_game
from run_local import run


class SnakeTests(unittest.TestCase):
    def fixture(self, body, direction, food, width=4, height=4):
        # Small handcrafted reachable positions test edge cases, not public reset options.
        g = make_game()
        g.width, g.height = width, height
        g._body, g._direction, g._food = list(body), direction, food
        return g

    def test_reverse_and_noop(self):
        g = self.fixture([(2, 1), (1, 1), (0, 1)], "RIGHT", (0, 0), 8, 8)
        g.tick("LEFT")
        self.assertEqual(g.observe()["body"][0], [3, 1])
        self.assertTrue(g.observe()["reverse_ignored"])
        g.tick(None)
        self.assertEqual(g.observe()["body"][0], [4, 1])
        self.assertEqual(g.ticks, 2)

    def test_growth_and_food(self):
        g = self.fixture([(2, 1), (1, 1), (0, 1)], "RIGHT", (3, 1))
        g.tick("RIGHT")
        obs = g.observe()
        self.assertEqual(obs["length"], 4)
        self.assertEqual(obs["score"], 1)
        self.assertAlmostEqual(g.score, 1 / 13)
        self.assertNotIn(obs["food"], obs["body"])

    def test_departing_tail_and_retained_body(self):
        body = [(1, 1), (1, 2), (0, 2), (0, 1)]
        g = self.fixture(body, "UP", (3, 3))
        g.tick("LEFT")
        self.assertFalse(g.done)
        self.assertEqual(g.observe()["body"][0], [0, 1])
        g = self.fixture(body + [(0, 0)], "UP", (3, 3))
        before = g.observe()["body"]
        g.tick("LEFT")
        self.assertEqual(g.observe()["end_reason"], "body_collision")
        self.assertEqual(g.observe()["body"], before)
        self.assertEqual(g.ticks, 1)

    def test_wall_and_terminal_freeze(self):
        g = self.fixture([(3, 1), (2, 1), (1, 1)], "RIGHT", (0, 0))
        before = g.observe()["body"]
        g.tick("NOOP")
        self.assertEqual(g.observe()["body"], before)
        self.assertEqual(g.observe()["attempted_destination"], [4, 1])
        self.assertEqual(g.observe()["end_reason"], "wall_collision")
        final = g.observe()
        g.tick("UP")
        g.truncate()
        self.assertEqual(g.observe(), final)

    def test_last_cell_win(self):
        cycle = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1),
                 (1, 2), (2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3),
                 (0, 2), (0, 1)]
        g = self.fixture(list(reversed(cycle[:-1])), "UP", cycle[-1])
        g.tick("UP")
        self.assertEqual(g.observe()["end_reason"], "board_full")
        self.assertIsNone(g.observe()["food"])
        self.assertEqual(g.observe()["length"], 16)
        self.assertEqual(g.score, 1.0)
        self.assertTrue(g.observe()["won"])

    def test_scenarios_and_snapshot_isolation(self):
        dimensions, headings = set(), set()
        for seed in range(128):
            g = make_game(profile="primary-v1")
            obs = g.reset(seed)
            self.assertEqual(obs, g.reset(seed))
            dimensions.add((g.width, g.height))
            headings.add(obs["direction"])
            self.assertEqual(len(set(map(tuple, obs["body"]))), 3)
            self.assertNotIn(obs["food"], obs["body"])
            dx, dy = DIRECTIONS[obs["direction"]]
            for a, b in zip(obs["body"], obs["body"][1:]):
                self.assertEqual((a[0] - b[0], a[1] - b[1]), (dx, dy))
            for _ in range(2):
                g.tick("NOOP")
                self.assertFalse(g.done)
            obs["body"][0][0] = -99
            self.assertNotEqual(g.observe()["body"][0][0], -99)
            self.assertLess(len(json.dumps(g.observe())), 200000)
        self.assertEqual(dimensions, {(8, 8), (10, 10), (12, 12), (16, 12)})
        self.assertEqual(headings, set(DIRECTIONS))

    def test_replay_random_games(self):
        for seed in range(16):
            a, b = make_game(profile="primary-v1"), make_game(profile="primary-v1")
            a.reset(seed)
            b.reset(seed)
            rng = random.Random(seed)
            for _ in range(200):
                action = rng.choice(a.action_spec.names)
                a.tick(action)
                b.tick(action)
                self.assertEqual(a.observe(), b.observe())
                obs = a.observe()
                self.assertEqual(obs["food_eaten"], obs["length"] - 3)
                self.assertNotIn(obs["food"], obs["body"])
                self.assertEqual(len(set(map(tuple, obs["body"]))), obs["length"])
                if a.done:
                    break
            self.assertTrue(a.done)
            c = make_game(profile="primary-v1")
            c.reset(seed)
            for record in a.replay()["actions"]:
                c.tick(record["applied"])
            for key in ("body", "food", "completion", "tick", "end_reason"):
                self.assertEqual(a.observe()[key], c.observe()[key])

    def test_real_seeded_full_board_game(self):
        # An independent test player follows a cycle only for a compatible start.
        for seed in range(128):
            g = make_game(profile="primary-v1")
            obs = g.reset(seed)
            if g.width != 8:
                continue
            w, h = g.width, g.height
            cycle = [(0, 0)]
            for y in range(h):
                xs = range(1, w) if y % 2 == 0 else range(w - 1, 0, -1)
                cycle.extend((x, y) for x in xs)
            cycle.extend((0, y) for y in range(h - 1, 0, -1))
            indices = {cell: i for i, cell in enumerate(cycle)}
            body = list(map(tuple, obs["body"]))
            head = indices[body[0]]
            if all(indices[cell] == (head - i) % len(cycle) for i, cell in enumerate(body)):
                break
        else:
            self.fail("No compatible development start")
        while not g.done and g.ticks < g.recommended_tick_cap:
            obs = g.observe()
            x, y = obs["body"][0]
            nx, ny = cycle[(indices[(x, y)] + 1) % len(cycle)]
            action = next(k for k, v in DIRECTIONS.items() if v == (nx - x, ny - y))
            g.tick(action)
        self.assertTrue(g.observe()["won"], g.info())
        self.assertEqual(g.score, 1.0)
        self.assertEqual(g.observe()["food_eaten"], 61)

    def test_external_cap_and_score_preservation(self):
        result = run(seed=0, fast=True, max_ticks=1)
        self.assertTrue(result["metrics"]["truncated"])
        self.assertEqual(result["metrics"]["ticks"], 1)
        g = self.fixture([(2, 1), (1, 1), (0, 1)], "RIGHT", (3, 1))
        g.tick("RIGHT")
        before = g.score
        g.truncate()
        self.assertEqual(g.score, before)
        self.assertTrue(g.done)
        self.assertFalse(g.observe()["terminated"])

    def test_invalid_inputs_and_profiles(self):
        g = make_game()
        for invalid in (True, 1.5, "3", None):
            with self.assertRaises(TypeError):
                g.reset(invalid)
        before = g.observe()
        for invalid in ("up", 0, {}, "move 1 2"):
            with self.assertRaises(ValueError):
                g.tick(invalid)
        self.assertEqual(before, g.observe())
        with self.assertRaises(ValueError):
            make_game(profile="unknown")
        for profile, (_, interval) in PROFILES.items():
            self.assertEqual(make_game(profile=profile).tick_s, interval)
        seeds = json.loads(Path(__file__).with_name("seeds.json").read_text())
        self.assertFalse(set(seeds["development"]) & set(seeds["held_out"]))

    def test_actual_runner_during_controller_stall(self):
        from metaharness.core.contract import Budget
        from metaharness.core.runner import run_episode
        from metaharness.models.base import TraceSink

        seen = []

        class Player:
            def run_episode(self, view):
                seen.append(view.ticks)
                time.sleep(1.05)  # No observations/actions while the wall ticker runs.
                seen.append(view.ticks)

        game = make_game()
        result = run_episode(types.SimpleNamespace(build=lambda ctx: Player()), game,
                             seed=0, system2=None, system1=None, trace=TraceSink(),
                             memory={}, budget=Budget(max_wall_s=5), clock_mode="wall",
                             harness_timeout_s=3)
        self.assertIsNone(result.error)
        self.assertEqual(seen[0], 0)
        self.assertGreaterEqual(seen[1], 3)
        self.assertEqual(result.steps, game.ticks)
        self.assertEqual(result.stale_ticks, game.ticks)


if __name__ == "__main__":
    unittest.main()
