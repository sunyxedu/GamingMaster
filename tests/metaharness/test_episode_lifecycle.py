"""Episode lifecycle: deadline and budget ends, cancellation of in-flight model calls, game
failures, adaptor close(), runner-counted steps, invalid trials and the virtual-clock scheduler."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from metaharness.core.contract import Budget
from metaharness.core.runner import evaluate_harness, preflight_harness
from metaharness.games.base import ActionSpec
from metaharness.models._cli import run_cli
from metaharness.models.base import ModelCall, System2Result
from metaharness.models.mock import ScriptedSystem2
from metaharness.testbeds.snake import SnakeGame

BASELINES = Path(__file__).resolve().parents[2] / "metaharness" / "baselines"


def _write(tmp_path, name, code):
    p = tmp_path / name
    p.write_text(code)
    return p


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/status") as fh:
            return "zombie" not in fh.read().lower()
    except OSError:
        return True


def _wait_dead(pid_file: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    pids = [int(x) for x in pid_file.read_text().split()]
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(_alive(p) for p in pids)


class SleepyCLI:
    """System-2 stand-in that runs a real subprocess through run_cli, like the codex/claude backends."""
    name = "sleepy"

    def __init__(self, pid_file: Path, sleep_s: float = 30.0) -> None:
        self.pid_file, self.sleep_s = pid_file, sleep_s

    def run(self, prompt, *, cwd=None, session_id=None, system=None, agentic=False, timeout_s=600.0,
            trace=None, images=None, full_access=False, web_search=False):
        t0 = time.time()
        code = (f"import os, time; open({str(self.pid_file)!r}, 'a').write(str(os.getpid()) + '\\n'); "
                f"time.sleep({self.sleep_s})")
        try:
            run_cli([sys.executable, "-c", code], cwd=None, stdin_text=None, timeout_s=timeout_s)
        except Exception as e:
            if trace:
                trace.log_call(ModelCall("system2", self.name, t0, time.time() - t0, {"prompt": prompt}, None,
                                         ok=False, error=str(e)))
            raise
        return System2Result(text="done", latency_s=time.time() - t0)


class CounterGame:
    name = "counter"
    action_spec = ActionSpec(names=["1"], descriptions={"1": "add one"})
    closes = 0

    def reset(self, seed):
        self.score, self.done = 0, False
        return self.observe()

    def observe(self):
        return {"score": self.score}

    def step(self, action):
        self.score += int(action)
        return self.observe()

    def info(self):
        return {}

    def describe(self):
        return "counter"

    def close(self):
        type(self).closes += 1


ACT_THEN_THINK = '''
def build(ctx):
    class H:
        def run_episode(self, game):
            ctx.act(1)
            ctx.think("slow")          # a CLI call that would take 30 s
            ctx.act(1)
    return H()
'''


def test_deadline_cancels_inflight_cli_call_and_keeps_score(tmp_path):
    pid_file = tmp_path / "pids"
    p = _write(tmp_path, "h.py", ACT_THEN_THINK)
    t0 = time.monotonic()
    r = evaluate_harness(p, CounterGame, [0], system2=SleepyCLI(pid_file), system1=None, clock_mode="wall",
                         budget_per_episode=Budget(max_wall_s=1.0), harness_timeout_s=1.0, grace_s=5.0)
    ep = r.episodes[0]
    assert ep.error is None and ep.ended_by == "deadline" and ep.score == 1
    assert time.monotonic() - t0 < 4
    assert _wait_dead(pid_file)


def test_harness_ignoring_the_deadline_is_an_error(tmp_path):
    p = _write(tmp_path, "h.py", '''
import time
def build(ctx):
    class H:
        def run_episode(self, game):
            while True:
                time.sleep(0.01)       # never looks at ctx
    return H()
''')
    r = evaluate_harness(p, CounterGame, [0], system2=None, system1=None, clock_mode="wall",
                         budget_per_episode=Budget(max_wall_s=0.3), harness_timeout_s=0.3, grace_s=0.3)
    ep = r.episodes[0]
    assert ep.ended_by == "deadline" and ep.error and "did not return within" in ep.error


def test_spawned_cli_call_is_cancelled_when_the_game_ends(tmp_path):
    pid_file = tmp_path / "pids"
    p = _write(tmp_path, "h.py", '''
def build(ctx):
    class H:
        def run_episode(self, game):
            ctx.spawn(lambda: ctx.think("replan"))     # still in flight when the game ends
            while not ctx.done:
                ctx.act("keep")
                ctx.wait_for_tick()
    return H()
''')
    t0 = time.monotonic()
    r = evaluate_harness(p, lambda **k: SnakeGame(tick_s=0.05, max_ticks=10, width=40, height=40, **k), [0],
                         system2=SleepyCLI(pid_file), system1=None, clock_mode="wall", harness_timeout_s=20)
    ep = r.episodes[0]
    assert ep.error is None and ep.ended_by == "game_done" and ep.info["pending_tasks"] == 0
    assert time.monotonic() - t0 < 5
    assert _wait_dead(pid_file)


def test_unfinished_background_python_work_is_an_error(tmp_path):
    p = _write(tmp_path, "h.py", '''
import time
def build(ctx):
    class H:
        def run_episode(self, game):
            ctx.spawn(lambda: time.sleep(3))       # ignores ctx.done
            ctx.act(1)
    return H()
''')
    r = evaluate_harness(p, CounterGame, [0], system2=None, system1=None, clock_mode="wall", grace_s=0.3)
    ep = r.episodes[0]
    assert ep.error and "still running" in ep.error


class BadTickSnake(SnakeGame):
    def tick(self, action):
        if action == "bad":
            raise ValueError("unknown action 'bad'")
        super().tick(action)


BAD_ACTION = '''
def build(ctx):
    class H:
        def run_episode(self, game):
            ctx.act("bad")
            while not ctx.done:
                ctx.wait_for_tick()
    return H()
'''


def test_tick_exception_ends_the_episode_as_a_game_error(tmp_path):
    p = _write(tmp_path, "h.py", BAD_ACTION)
    for clock in ("wall", "virtual"):
        t0 = time.monotonic()
        r = evaluate_harness(p, lambda **k: BadTickSnake(tick_s=0.05, max_ticks=1000, **k), [0], system2=None,
                             system1=None, clock_mode=clock, harness_timeout_s=10)
        ep = r.episodes[0]
        assert ep.ended_by == "game_error" and "tick (action 'bad')" in ep.error and "unknown action" in ep.error
        assert time.monotonic() - t0 < 3, clock          # ends at once instead of burning the deadline


def test_turn_steps_are_counted_by_the_runner(tmp_path):
    p = _write(tmp_path, "h.py", "def build(ctx):\n    class H:\n        def run_episode(self, game):\n"
                                 "            for _ in range(7): ctx.act(1)\n    return H()\n")
    r = evaluate_harness(p, CounterGame, [0, 1], system2=None, system1=None, clock_mode="virtual")
    assert [e.steps for e in r.episodes] == [7, 7] and r.to_json()["mean_steps"] == 7


def test_game_is_closed_after_every_episode(tmp_path):
    p = _write(tmp_path, "h.py", "def build(ctx):\n    class H:\n        def run_episode(self, game):\n"
                                 "            ctx.act(1)\n    return H()\n")
    CounterGame.closes = 0
    evaluate_harness(p, CounterGame, [0, 1, 2], system2=None, system1=None, clock_mode="virtual")
    assert CounterGame.closes == 3


def test_game_failures_outside_reset_are_episode_errors(tmp_path):
    p = _write(tmp_path, "h.py", "def build(ctx):\n    class H:\n        def run_episode(self, game):\n"
                                 "            ctx.act(1)\n    return H()\n")
    made = []

    def factory(**kw):
        made.append(1)
        if len(made) == 1:
            raise ConnectionError("server is starting")
        return CounterGame()

    class BrokenScore(CounterGame):
        def reset(self, seed):
            self.n, self.done = 0, False

        def observe(self):
            return {"n": self.n}

        def step(self, action):
            self.n += 1
            return self.observe()

        @property
        def score(self):
            raise RuntimeError("bridge disconnected")

    r = evaluate_harness(p, factory, [0, 1], system2=None, system1=None, clock_mode="virtual")
    assert r.fatal is None and "game construction failed" in r.episodes[0].error and r.episodes[1].error is None
    r = evaluate_harness(p, BrokenScore, [0], system2=None, system1=None, clock_mode="virtual")
    ep = r.episodes[0]
    assert r.fatal is None and ep.ended_by == "game_error" and "bridge disconnected" in ep.error


def test_invalid_trial_is_replayed_once_then_an_error(tmp_path):
    p = _write(tmp_path, "h.py", "def build(ctx):\n    class H:\n        def run_episode(self, game):\n"
                                 "            ctx.act(1)\n    return H()\n")
    made = []

    class Flaky(CounterGame):
        def __init__(self):
            made.append(1)
            self.first = len(made) == 1

        def info(self):
            return {"valid_trial": not self.first, "invalid_reason": "server lagged"}

    r = evaluate_harness(p, Flaky, [0], system2=None, system1=None, clock_mode="virtual", out_dir=tmp_path / "e")
    assert r.episodes[0].error is None and r.metadata["invalid_trials"] == [{"seed": 0, "reason": "server lagged"}]
    assert (tmp_path / "e" / "trace_seed0.invalid0.jsonl").exists()

    class Broken(CounterGame):
        def info(self):
            return {"valid_trial": False, "invalid_reason": "server down"}

    r = evaluate_harness(p, Broken, [0], system2=None, system1=None, clock_mode="virtual")
    assert "invalid trial" in r.episodes[0].error


def test_budget_reports_the_enforced_deadline(tmp_path):
    p = _write(tmp_path, "h.py", '''
def build(ctx):
    class H:
        def run_episode(self, game):
            ctx.act(int(ctx.budget.remaining()["wall_s"] * 100))
    return H()
''')
    r = evaluate_harness(p, CounterGame, [0], system2=None, system1=None, clock_mode="wall",
                         budget_per_episode=Budget(max_wall_s=900), harness_timeout_s=2.0)
    assert 150 < r.episodes[0].score <= 200 and r.metadata["deadline_s"] == 2.0


BACKGROUND_PLANNER = '''
def build(ctx):
    class H:
        def run_episode(self, game):
            def planner():
                while not ctx.done:
                    ctx.think("plan")             # 1 virtual second each
            ctx.spawn(planner)
            while not ctx.done:
                ctx.act("keep")
                ctx.wait_for_tick()
    return H()
'''


def test_virtual_background_calls_are_not_starved_and_deterministic(tmp_path):
    p = _write(tmp_path, "h.py", BACKGROUND_PLANNER)
    counts = set()
    for _ in range(3):
        r = evaluate_harness(p, lambda **k: SnakeGame(width=200, height=12, max_ticks=100, **k), [0],
                             system2=ScriptedSystem2(latency_s=1.0), system1=None, clock_mode="virtual",
                             budget_per_episode=Budget(max_system2_calls=100), harness_timeout_s=30)
        ep = r.episodes[0]
        assert ep.error is None and ep.ended_by == "game_done", ep.error
        counts.add(ep.system2_calls)
    # 100 ticks x 0.25 s = 25 virtual seconds of back-to-back 1 s calls
    assert len(counts) == 1 and counts.pop() in (25, 26)


def test_preflight_of_the_reference_harness_is_deterministic(tmp_path):
    verdicts = [preflight_harness(BASELINES / "tier2_reference.py", lambda **k: SnakeGame(**k), seed=s)
                for s in range(6)]
    assert all(v["ok"] for v in verdicts), [v["error"] for v in verdicts if not v["ok"]]
    assert all(v["coverage"]["system1_answers"] > 0 for v in verdicts)


class EndlessGame:
    name = "endless"
    action_spec = ActionSpec(names=["a", "noop"], noop="noop")
    tick_s = 0.1

    def reset(self, seed):
        self.ticks, self.done, self.score = 0, False, 0.0

    def tick(self, action):
        self.ticks += 1

    def observe(self):
        return {"t": self.ticks}

    def info(self):
        return {}

    def describe(self):
        return "never ends"


def test_preflight_caps_ticks_and_reports_coverage(tmp_path):
    p = _write(tmp_path, "h.py", "def build(ctx):\n    class H:\n        def run_episode(self, game):\n"
                                 "            while not ctx.done: ctx.wait_for_tick()\n    return H()\n")
    t0 = time.monotonic()
    rep = preflight_harness(p, EndlessGame, tier2=True, max_ticks=500)
    assert rep["ok"] and rep["ended_by"] == "virtual_cap" and rep["steps"] <= 500
    assert time.monotonic() - t0 < 10
    assert any("never posted an action" in w for w in rep["warnings"])


def test_live_action_spec_is_visible_to_the_harness(tmp_path):
    from metaharness.core.runner import _GameView
    import threading

    g: Any = CounterGame()
    view = _GameView(g, threading.Event())
    g.action_spec = ActionSpec(names=["1", "2"])
    assert list(view.action_spec.names) == ["1", "2"]
