"""Short public-interface fixtures; no engine or model backend is imported."""
import json
import math
import re
import unittest
from types import SimpleNamespace

import harness as h


class Game:
    def __init__(self, goal=999):
        self.done = False
        self.score = 0
        self.n = 0
        self.goal = goal
        self.action_spec = SimpleNamespace(
            names=["noop", "reset", "up", "click 32 32"], noop="noop",
            descriptions={"click 32 32": "click X Y; integer coordinates 0..63"}, notes="")
        self.obs = {"renamed_pixels": [[0, 1], [2, 3]],
                    "renamed_phase": 0, "status_alias": "playing"}

    def describe(self):
        return "Synthetic turn game. Use currently listed actions."

    def observe(self):
        return self.obs

    def step(self, action):
        click = re.fullmatch(r"click (\d+) (\d+)", action)
        if action not in self.action_spec.names:
            if not click or not all(0 <= int(c) <= 63 for c in click.groups()):
                raise ValueError("illegal stand-in action")
        self.n += 1
        # Mutate the same dictionary to check snapshot fidelity.
        self.obs["renamed_pixels"][0][0] = self.n % 16
        if self.n >= self.goal:
            self.score = 1
            self.done = True
        return self.obs


class Context:
    def __init__(self, reply="up", wall=900, latency=25, calls=40, goal=999):
        self.game = Game(goal)
        self.t = 0.0
        self.clock = SimpleNamespace(now=lambda: self.t)
        self.wall = wall
        self.max_calls = calls
        self.calls = 0
        self.budget = SimpleNamespace(remaining=self.remaining)
        self.reply = reply
        self.latency = latency
        self.memory = {}
        self.events = []
        self.actions = []
        self.timeouts = []
        self.system1 = None

    def remaining(self):
        return {"system2_calls": self.max_calls - self.calls,
                "system1_calls": 0,
                "wall_s": None if self.wall is None else self.wall - self.t}

    @property
    def done(self):
        return self.game.done

    def observe(self):
        return self.game.observe()

    def think(self, prompt, *, session_id=None, system=None, images=None, timeout_s=300):
        self.calls += 1
        self.timeouts.append(timeout_s)
        if isinstance(self.reply, BaseException):
            raise self.reply
        latency = self.latency(self.calls) if callable(self.latency) else self.latency
        if latency > timeout_s:
            self.t += timeout_s
            raise TimeoutError("stand-in request timeout")
        self.t += latency
        reply = self.reply(self.calls) if callable(self.reply) else self.reply
        return SimpleNamespace(text=reply, session_id="fixture")

    def act(self, action):
        self.actions.append(action)
        self.t += 0.001
        return self.game.step(action)

    def log(self, **fields):
        self.events.append(fields)

    def run(self):
        h.build(self).run_episode(self.game)
        return self


class HarnessTests(unittest.TestCase):
    def test_too_late_to_admit(self):
        for wall in (0, 3.7656729221343994, 36.6256148815155, 70):
            ctx = Context(wall=wall).run()
            self.assertEqual(ctx.calls, 0)
            self.assertEqual(ctx.actions, [])

    def test_normal_deadline_returns_with_score(self):
        ctx = Context().run()
        self.assertLess(ctx.t, 900)
        self.assertGreaterEqual(900 - ctx.t, 10)
        self.assertEqual(ctx.calls, 34)
        self.assertTrue(any(e["event"] == "deadline_stop" for e in ctx.events))

    def test_latency_reserve_adapts(self):
        ctx = Context(wall=200, latency=80).run()
        self.assertEqual(ctx.calls, 1)
        self.assertAlmostEqual(ctx.timeouts[0], 190)
        self.assertEqual(len(ctx.actions), 1)

    def test_unexpected_latency_times_out_cleanly(self):
        ctx = Context(wall=100, latency=1000).run()
        self.assertEqual(ctx.t, 90)
        self.assertEqual(ctx.actions, [])
        self.assertEqual(ctx.events[-1]["event"], "model_timeout_stop")

    def test_model_unavailability_propagates(self):
        for base in (RuntimeError, TimeoutError):
            class ModelUnavailable(base):
                pass
            error = ModelUnavailable("offline stand-in")
            with self.assertRaises(ModelUnavailable) as caught:
                Context(reply=error).run()
            self.assertIs(caught.exception, error)

    def test_unlimited_time_respects_call_cap(self):
        for wall in (None, math.inf):
            ctx = Context(wall=wall, latency=1).run()
            self.assertEqual(ctx.calls, 40)

    def test_stale_budget_and_prompt_time(self):
        ctx = Context(wall=100)
        ctx.budget.remaining = lambda: {"wall_s": 100, "system2_calls": 40}
        deadline = h.Deadline(ctx)
        self.assertEqual(deadline.admit(), 90)
        ctx.t = 40
        self.assertIsNone(deadline.admit())

    def test_garbage_replies(self):
        replies = [None, "", "first option", "{broken", "[]", "null", "true",
                   "{\"plan\":null}", "{\"plan\":[null,5,{},true]}",
                   "{\"schema\":{\"recovery\":{\"path\":[],\"action\":{}}}}",
                   "{\"schema\":{\"phase_paths\":[null,[],{}]}}",
                   "{\"plan\":[{\"action\":\"up\",\"repeat\":{},\"expect\":null}]}",
                   "{\"plan\":[{\"action\":\"up\",\"expect\":{\"pixels\":[[{},0,0,1],[]]}}]}",
                   "[" * 10000, "click " + "9" * 5000 + " 0"]
        for reply in replies:
            with self.subTest(reply=str(reply)[:80]):
                ctx = Context(reply=reply, latency=0, calls=2).run()
                self.assertEqual(ctx.calls, 2)
                self.assertTrue(ctx.actions)
                self.assertFalse(any(e["event"] == "invalid_action" for e in ctx.events))

    def test_click_grammar_and_fences(self):
        ctx = Context(reply='```json\n{"plan":["click 19 20"]}\n```', calls=1).run()
        self.assertEqual(ctx.actions, ["click 19 20"])
        actions = h.Actions(ctx.game, ctx.game.describe())
        for invalid in ("click", "click 64 0", "click -1 0", "click 1.5 0"):
            self.assertIsNone(actions.validate(invalid))

    def test_phase_and_prediction_stop_batches(self):
        for expect in ({"pixels": [["/renamed_pixels", 0, 0, 99]]},
                       {"equals": {"/renamed_phase": 99}}):
            reply = json.dumps({"mode": "execute", "plan": [
                {"action": "up", "repeat": 16, "expect": expect}]})
            ctx = Context(reply=reply, calls=1).run()
            self.assertEqual(len(ctx.actions), 1)
            event = next(e for e in ctx.events if e["event"] == "measured_transition")
            self.assertTrue(event["refuted"])
            self.assertEqual(event["delta"]["grids"]["/renamed_pixels"]["pixels_changed"], 1)
        ctx = Context(reply='{"mode":"execute","plan":[{"action":"up","repeat":16}]}',
                      calls=1, goal=1).run()
        self.assertEqual(ctx.actions, ["up"])
        self.assertEqual(ctx.game.score, 1)

    def test_time_guard_inside_batch(self):
        ctx = Context(reply='{"mode":"execute","plan":[{"action":"up","repeat":16}]}',
                      wall=100, latency=25)
        original_act = ctx.act
        def slow_action(action):
            result = original_act(action)
            ctx.t += 30
            return result
        ctx.act = slow_action
        ctx.run()
        self.assertEqual(len(ctx.actions), 3)
        # A synchronous game action cannot be preempted, but the next is not sent.
        self.assertEqual(ctx.calls, 1)

    def test_new_episode_clears_candidate_memory(self):
        ctx = Context(calls=1).run()
        old = ctx.memory["candidate_0002"]
        ctx.calls = 0
        ctx.run()
        self.assertIsNot(old, ctx.memory["candidate_0002"])


if __name__ == "__main__":
    unittest.main()
