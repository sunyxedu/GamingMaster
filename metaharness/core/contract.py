"""The harness contract — the *only* thing a proposed harness must satisfy.

A harness is a Python module that defines::

    def build(ctx: HarnessContext) -> HarnessProgram

and ``HarnessProgram`` has one method::

    def run_episode(self, game) -> None

The harness owns the whole control loop for one episode. Everything it may
use comes through ``ctx``:

* ``ctx.system2``   — Astra (slow, reasoning). ``ctx.think(prompt, session=...)``
                      wraps it with tracing + budget accounting.
* ``ctx.system1``   — Jev (fast, typed decisions). ``ctx.decide(state, questions)``
                      wraps it with tracing + latency accounting. Tier 1 tasks
                      pass ``system1=None``.
* ``ctx.spawn(fn)`` — run ``fn`` on a background thread (e.g. an Astra call
                      while Jev keeps the game alive). Returns a Future.
* ``ctx.clock``     — the game clock. In *virtual* mode, model latency is
                      charged to the clock deterministically (tests); in
                      *wall* mode it is real time.
* ``ctx.memory``    — a dict that persists across episodes of the same
                      evaluation (the harness may learn between seeds).
* ``ctx.log(...)``  — free-form events into the trace.
* ``ctx.budget``    — remaining System-2 calls / tokens; exceeding it ends the
                      episode (score kept), so harnesses must be thrifty.

Real-time games (tier 2) expose ``game.tick_s``. The runner keeps ticking the
game on its own clock: in wall mode a thread ticks every ``tick_s``; in
virtual mode each model call's latency advances the game by
``latency / tick_s`` ticks (using the *last action the harness posted*).
The harness posts actions with ``ctx.act(action)``; posting is instant
(virtual mode charges a 1 ms overhead so a loop that decides without any
model call still lets time pass). ``ctx.wait_for_tick()`` blocks until the
next tick boundary — use it instead of spinning.
Turn games (tier 1) are stepped with ``ctx.act(action)`` synchronously and
return the next observation.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from ..models.base import ModelUnavailable, System1Answer, System1Model, System2Model, System2Result, TraceSink


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_system2_calls: int = 50
    max_system1_calls: int = 5000
    max_wall_s: float = 900.0
    system2_calls: int = 0
    system1_calls: int = 0
    t_start: float = field(default_factory=time.time)

    def check(self) -> None:
        if self.system2_calls > self.max_system2_calls:
            raise BudgetExceeded(f"system2 calls > {self.max_system2_calls}")
        if self.system1_calls > self.max_system1_calls:
            raise BudgetExceeded(f"system1 calls > {self.max_system1_calls}")
        if time.time() - self.t_start > self.max_wall_s:
            raise BudgetExceeded(f"wall time > {self.max_wall_s}s")

    def remaining(self) -> dict[str, float]:
        return {
            "system2_calls": self.max_system2_calls - self.system2_calls,
            "system1_calls": self.max_system1_calls - self.system1_calls,
            "wall_s": self.max_wall_s - (time.time() - self.t_start),
        }


class Clock(Protocol):
    mode: str

    def now(self) -> float: ...
    def charge(self, seconds: float) -> None: ...
    def sleep(self, seconds: float) -> None: ...


class WallClock:
    mode = "wall"

    def now(self) -> float:
        return time.time()

    def charge(self, seconds: float) -> None:  # real latency already elapsed
        pass

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class VirtualClock:
    """Deterministic discrete-event clock.

    The *driver* thread (the harness's ``run_episode`` thread) advances time
    when it is charged (model latency, sleeps); ``on_advance(t0, t1)`` lets the
    runner apply game ticks that fall inside the interval. A charge from any
    *other* thread (``ctx.spawn`` workers — e.g. a background Astra call) does
    not advance time: it blocks until the driver's clock reaches the wake time,
    so a 2 s Astra call overlaps with 2 s worth of Jev calls exactly as it
    would in wall time. If the driver is idle (waiting on that future) for
    ``idle_s`` real seconds, the earliest waiter advances the clock itself.
    """
    mode = "virtual"

    def __init__(self, idle_s: float = 0.05) -> None:
        self._t = 0.0
        self.on_advance: Optional[Callable[[float, float], None]] = None
        self._cv = threading.Condition()
        self._driver: Optional[int] = None
        self._last_driver_charge = time.monotonic()
        self.idle_s = idle_s

    def set_driver(self, ident: Optional[int] = None) -> None:
        self._driver = ident if ident is not None else threading.get_ident()

    def now(self) -> float:
        return self._t

    def _advance_to(self, t1: float) -> None:
        # called with the condition held
        t0 = self._t
        if t1 <= t0:
            return
        self._t = t1
        cb = self.on_advance
        self._cv.notify_all()
        if cb:
            self._cv.release()
            try:
                cb(t0, t1)
            finally:
                self._cv.acquire()

    def charge(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        me = threading.get_ident()
        with self._cv:
            if self._driver is None or me == self._driver:
                self._last_driver_charge = time.monotonic()
                self._advance_to(self._t + seconds)
                return
            wake = self._t + seconds
            while self._t < wake:
                self._cv.wait(timeout=self.idle_s)
                if self._t >= wake:
                    break
                if time.monotonic() - self._last_driver_charge >= self.idle_s:
                    # driver is idle (blocked on us) — the earliest waiter moves time
                    self._last_driver_charge = time.monotonic()
                    self._advance_to(wake)

    def sleep(self, seconds: float) -> None:
        self.charge(seconds)


class HarnessContext:
    def __init__(
        self,
        *,
        system2: Optional[System2Model],
        system1: Optional[System1Model],
        trace: TraceSink,
        clock: Clock,
        budget: Budget,
        memory: dict[str, Any],
        game_kind: str,                 # "turn" | "realtime"
        act_fn: Callable[[Any], Any],
        observe_fn: Callable[[], Any],
        done_fn: Callable[[], bool],
        latency_override: Optional[Mapping[str, float]] = None,
        tick_s: Optional[float] = None,
        act_overhead_s: float = 0.001,
    ) -> None:
        self.tick_s = tick_s
        self.act_overhead_s = act_overhead_s
        self.system2 = system2
        self.system1 = system1
        self.trace = trace
        self.clock = clock
        self.budget = budget
        self.memory = memory
        self.game_kind = game_kind
        self._act = act_fn
        self._observe = observe_fn
        self._done = done_fn
        self._latency_override = dict(latency_override or {})
        self._pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="harness")
        self.last_system2: Optional[System2Result] = None
        self.errors: list[str] = []

    # -- game -----------------------------------------------------------
    def act(self, action: Any) -> Any:
        """Realtime: post the current action (instant). Turn: step and return obs."""
        out = self._act(action)
        if self.game_kind == "realtime" and self.clock.mode == "virtual":
            self.clock.charge(self.act_overhead_s)
        return out

    def wait_for_tick(self) -> None:
        """Block until the next game tick boundary (realtime games only)."""
        if self.game_kind != "realtime" or not self.tick_s:
            return
        now = self.clock.now()
        nxt = (int(now / self.tick_s) + 1) * self.tick_s
        self.clock.sleep(max(1e-6, nxt - now))

    def observe(self) -> Any:
        return self._observe()

    @property
    def done(self) -> bool:
        return self._done()

    # -- models ---------------------------------------------------------
    def think(self, prompt: str, *, session_id: Optional[str] = None, system: Optional[str] = None,
              timeout_s: float = 300.0) -> System2Result:
        if self.system2 is None:
            raise RuntimeError("no System-2 model in this task")
        self.budget.system2_calls += 1
        self.budget.check()
        res = self.system2.run(prompt, session_id=session_id, system=system, agentic=False,
                               timeout_s=timeout_s, trace=self.trace)
        charged = self._latency_override.get("system2", res.latency_s)
        self.clock.charge(charged)
        self.last_system2 = res
        return res

    def decide(self, state: Any, questions: Mapping[str, Any], *, timeout_s: float = 5.0) -> dict[str, System1Answer]:
        if self.system1 is None:
            raise RuntimeError("no System-1 model in this task")
        self.budget.system1_calls += 1
        self.budget.check()
        t0 = time.time()
        ans = self.system1.decide(state, questions, timeout_s=timeout_s, trace=self.trace)
        lat = time.time() - t0
        charged = self._latency_override.get("system1", getattr(self.system1, "latency_s", lat))
        if self.clock.mode == "virtual":
            self.clock.charge(charged)
        return ans

    def spawn(self, fn: Callable[[], Any]) -> cf.Future:
        return self._pool.submit(fn)

    def log(self, **event: Any) -> None:
        event.setdefault("type", "harness")
        event["clock"] = self.clock.now()
        self.trace.write(event)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class HarnessProgram(Protocol):
    def run_episode(self, game: Any) -> None: ...


def load_harness_module(path: Path | str, name: Optional[str] = None):
    """Import a harness file as a fresh module (each candidate isolated)."""
    path = Path(path)
    mod_name = name or f"_harness_{abs(hash(str(path.resolve())))}_{int(time.time()*1000)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load harness at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build"):
        raise ImportError(f"harness {path} defines no build(ctx)")
    return module


__all__ = ["Budget", "BudgetExceeded", "Clock", "WallClock", "VirtualClock", "HarnessContext",
           "HarnessProgram", "load_harness_module", "ModelUnavailable"]
