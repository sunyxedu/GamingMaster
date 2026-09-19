"""Virtual clock semantics and runner latency accounting."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from metaharness.core.contract import Budget, VirtualClock
from metaharness.core.runner import evaluate_harness
from metaharness.games.snake import SnakeGame
from metaharness.models.mock import HeuristicSystem1, ScriptedSystem2

HARNESS_DIR = Path(__file__).parent / "harnesses"


def test_virtual_clock_background_charge_overlaps_driver():
    clk = VirtualClock(idle_s=0.02)
    advances = []
    clk.on_advance = lambda a, b: advances.append((a, b))
    clk.set_driver()
    done = threading.Event()

    def worker():
        clk.charge(2.0)      # must not advance time by itself while driver is active
        done.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(0.005)
    assert clk.now() == 0.0
    for _ in range(5):
        clk.charge(0.3)
    assert abs(clk.now() - 1.5) < 1e-9 and not done.is_set()
    clk.charge(0.6)          # crosses the worker's wake time (2.0)
    assert done.wait(1.0)
    assert abs(clk.now() - 2.1) < 1e-9
    assert advances[0] == (0.0, 0.3)


def test_virtual_clock_idle_driver_lets_waiter_advance():
    clk = VirtualClock(idle_s=0.02)
    clk.set_driver()
    t0 = time.time()
    res = []

    def worker():
        clk.charge(5.0)
        res.append(clk.now())

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(2.0)
    assert res == [5.0] and time.time() - t0 < 1.0     # virtual, not real, seconds


def _write(tmp_path, name, code):
    p = tmp_path / name
    p.write_text(code)
    return p


SPIN_HARNESS = '''
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        while not self.ctx.done:
            self.ctx.act("up")            # never calls a model, never waits
def build(ctx): return H(ctx)
'''

TICK_HARNESS = '''
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        while not self.ctx.done:
            obs = game.observe()
            self.ctx.decide(obs, {"m": {"type": "choice", "instructions": "q", "criteria": {"up": "", "down": ""}}})
            self.ctx.act("keep")
            self.ctx.wait_for_tick()
def build(ctx): return H(ctx)
'''


def test_spinning_harness_terminates_in_virtual_mode(tmp_path):
    p = _write(tmp_path, "spin.py", SPIN_HARNESS)
    r = evaluate_harness(p, lambda **k: SnakeGame(max_ticks=50, **k), [0], system2=None, system1=None,
                         clock_mode="virtual", harness_timeout_s=20)
    ep = r.episodes[0]
    assert ep.error is None and ep.steps >= 1 and ep.stale_ticks + ep.decided_ticks == ep.steps


def test_stale_tick_accounting_with_slow_system1(tmp_path):
    p = _write(tmp_path, "tick.py", TICK_HARNESS)
    s1 = HeuristicSystem1(latency_s=0.6)   # slower than the 0.25 s tick
    r = evaluate_harness(p, lambda **k: SnakeGame(max_ticks=40, **k), [0, 1], system2=None, system1=s1,
                         clock_mode="virtual", harness_timeout_s=20)
    assert r.fatal is None
    for ep in r.episodes:
        assert ep.system1_calls > 0
        assert ep.stale_ticks > ep.decided_ticks          # most ticks ran on a stale action
        assert ep.latency_s1 and abs(ep.latency_s1[0] - 0.6) < 1e-9
    assert r.stale_fraction is not None and r.stale_fraction > 0.5


def test_budget_exceeded_is_reported_not_fatal(tmp_path):
    code = '''
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        while not self.ctx.done:
            self.ctx.think("plan?")
            self.ctx.act("keep")
def build(ctx): return H(ctx)
'''
    p = _write(tmp_path, "b.py", code)
    r = evaluate_harness(p, lambda **k: SnakeGame(max_ticks=1000, **k), [0], system2=ScriptedSystem2(latency_s=0.1),
                         system1=None, clock_mode="virtual", budget_per_episode=Budget(max_system2_calls=3))
    ep = r.episodes[0]
    assert ep.error and "budget" in ep.error and ep.system2_calls == 3


def test_load_error_is_fatal(tmp_path):
    p = _write(tmp_path, "bad.py", "def not_build(): pass\n")
    r = evaluate_harness(p, lambda **k: SnakeGame(**k), [0], system2=None, system1=None, clock_mode="virtual")
    assert r.fatal and "build" in r.fatal and r.episodes == []


def test_wall_clock_realtime_ticks_without_harness(tmp_path):
    code = '''
import time
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        time.sleep(0.5)     # do nothing: the game must keep ticking on its own
def build(ctx): return H(ctx)
'''
    p = _write(tmp_path, "idle.py", code)
    r = evaluate_harness(p, lambda **k: SnakeGame(tick_s=0.05, max_ticks=1000, **k), [0], system2=None,
                         system1=None, clock_mode="wall", harness_timeout_s=5)
    ep = r.episodes[0]
    assert ep.steps >= 5 and ep.stale_ticks == ep.steps


def test_wall_ticker_is_phase_aligned_with_wait_for_tick(tmp_path):
    code = '''
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        while not self.ctx.done:
            self.ctx.act("keep")
            self.ctx.wait_for_tick()
def build(ctx): return H(ctx)
'''
    p = _write(tmp_path, "phase.py", code)
    r = evaluate_harness(p, lambda **k: SnakeGame(tick_s=0.05, max_ticks=30, width=120, height=120, **k), [0],
                         system2=None, system1=None, clock_mode="wall", harness_timeout_s=10)
    ep = r.episodes[0]
    # one fresh decision per tick: (almost) no stale ticks and no doubled decisions
    assert ep.steps >= 20 and ep.stale_ticks <= 2
