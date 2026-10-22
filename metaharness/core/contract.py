"""The harness contract — the *only* thing a proposed harness must satisfy.

A harness is a Python module that defines::

    def build(ctx: HarnessContext) -> HarnessProgram

and ``HarnessProgram`` has one method::

    def run_episode(self, game) -> None

The harness owns the whole control loop for one episode. Everything it may
use comes through ``ctx``:

* ``ctx.think(prompt, session_id=...)`` — System-2 (Astra: slow, reasoning), traced and budgeted.
* ``ctx.decide(state, questions)``      — System-1 (Jev: fast typed judgments), traced and budgeted.
                                          Tier-1 tasks have no System-1 (``ctx.system1`` is None).
* ``ctx.spawn(fn)`` — run ``fn`` on a background thread (e.g. an Astra call while Jev keeps the
                      game alive). Returns a ``concurrent.futures.Future``.
* ``ctx.clock``     — the game clock. In *virtual* mode model latency is charged to the clock
                      deterministically (tests, pre-flight); in *wall* mode it is real time.
* ``ctx.memory``    — a dict, fresh every episode unless the evaluation explicitly shares it
                      across seeds (the evaluation manifest records which protocol was used).
* ``ctx.log(...)``  — free-form events into the trace.
* ``ctx.budget``    — what is left of this episode's call budgets and wall-clock deadline.

Real-time games (tier 2) expose ``game.tick_s`` and tick on their own clock: in wall mode a thread
applies a tick every ``tick_s``; in virtual mode a model call's latency advances the game by
``latency / tick_s`` ticks. ``ctx.act(action)`` posts the current action (instant; virtual mode
charges 1 ms so a loop that never calls a model still lets time pass). Ticks apply the noop until
the first action is posted; afterwards a tick without a fresh action repeats the last posted
action. ``ctx.wait_for_tick()`` blocks until the game has applied its next tick — use it instead
of spinning. Turn games (tier 1) are stepped by ``ctx.act(action)``, which returns the next
observation.

Exact signatures (use these; anything else is a contract error caught in pre-flight)::

    ctx.think(prompt: str, *, session_id: str | None = None, system: str | None = None,
              images: list[str] | None = None, timeout_s: float = 300.0) -> System2Result   # .text .session_id .latency_s
    ctx.decide(state: Any, questions: dict[str, Choice | Score | Noul], *, timeout_s: float = 5.0)
              -> dict[str, System1Answer]                                                   # .value .probabilities .confidence
    ctx.act(action) -> observation | None      ctx.observe() -> observation      ctx.done -> bool
    ctx.wait_for_tick() -> None                ctx.spawn(fn) -> concurrent.futures.Future
    ctx.log(*args, **fields) -> None           ctx.memory: dict     ctx.clock.now() -> float (s)
    ctx.budget.remaining() -> {"system2_calls": int, "system1_calls": int, "wall_s": float}
    game.describe() -> str    game.action_spec.names / .descriptions / .noop / .notes    game.observe()
    game.tick_s (realtime only)    game.ticks    game.score    game.done

What ctx calls raise (``from metaharness.core.contract import BudgetExceeded, EpisodeOver``)::

    TimeoutError      the model call exceeded its timeout_s — a slow answer is a normal event
    RuntimeError      any other failed model call (bad response, transient backend error)
    BudgetExceeded    (a RuntimeError) a call budget is used up or the episode's wall-clock deadline
                      has passed; .kind is "system2" | "system1" | "wall"
    EpisodeOver       (a BudgetExceeded) the episode is over — the game ended, the deadline passed or
                      the evaluator closed it — before or while the call ran
    ModelUnavailable  the backend is out of quota or logged out; the run stops. The context records
                      it even if the harness catches it, so ``except Exception`` around a call is safe.
    ctx.act on a turn game raises whatever the game raises for an illegal action (catch it to recover);
    in a real-time game an action that makes ``game.tick`` raise ends the episode as a game error.

How an episode ends — the score reached is always kept:

* the game is over, or ``run_episode`` returns;
* a call budget is used up, or the wall-clock deadline (``ctx.budget.remaining()["wall_s"]``)
  passes: from then on ``ctx.done`` is True for the deadline and ctx calls raise ``BudgetExceeded``
  or ``EpisodeOver``. Letting that exception propagate out of ``run_episode`` is a normal end.
  After the deadline the harness has a short grace period to return.
* Model calls still in flight when the game ends or the episode closes are cancelled (they raise
  ``EpisodeOver``); a harness may return without joining its ``ctx.spawn`` futures.

A candidate is eligible for selection only if none of its training episodes ends with an *error*:
any other exception escaping ``run_episode`` (e.g. a TimeoutError the harness did not catch), a
harness that does not return within the grace period after the deadline, ``ctx.spawn`` work that
is still running a few seconds after ``run_episode`` returned (plain Python loops must stop when
``ctx.done``), or a failure of the game itself.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import importlib.util
import itertools
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol

from ..models._cli import cancel_scope
from ..models.base import (ModelCallCancelled, ModelUnavailable, System1Answer, System1Model, System2Model,
                           System2Result, TraceSink)


class BudgetExceeded(RuntimeError):
    """A per-episode budget refused the request. ``kind``: "system2" / "system1" (that call budget
    is used up), "wall" (the deadline passed) or "closed" (the episode is over, see EpisodeOver)."""

    def __init__(self, message: str = "budget exceeded", kind: str = "wall") -> None:
        super().__init__(message)
        self.kind = kind


class EpisodeOver(BudgetExceeded):
    """The episode is over (game ended, deadline passed or the evaluator closed it)."""

    def __init__(self, message: str = "episode is over") -> None:
        super().__init__(message, kind="closed")


@dataclass
class Budget:
    max_system2_calls: int = 50
    max_system1_calls: int = 5000
    max_wall_s: float = 900.0
    system2_calls: int = 0
    system1_calls: int = 0
    t_start: float = field(default_factory=time.monotonic)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        """Start the wall-clock deadline now (the evaluator calls this once the game is reset, so
        slow resets — server start, world generation — do not shorten the harness's time)."""
        self.t_start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t_start

    def reserve(self, kind: str) -> None:
        """Count only requests admitted to the backend, atomically across workers."""
        with self._lock:
            self.check()
            attr = kind + "_calls"
            limit = getattr(self, "max_" + attr)
            if getattr(self, attr) >= limit:
                raise BudgetExceeded(f"{kind} call budget exhausted ({limit} calls)", kind=kind)
            setattr(self, attr, getattr(self, attr) + 1)

    def check(self) -> None:
        if self.elapsed() > self.max_wall_s:
            raise BudgetExceeded(f"wall time > {self.max_wall_s}s", kind="wall")

    def remaining(self) -> dict[str, float]:
        return {
            "system2_calls": self.max_system2_calls - self.system2_calls,
            "system1_calls": self.max_system1_calls - self.system1_calls,
            "wall_s": max(0.0, self.max_wall_s - self.elapsed()),
        }


class Clock(Protocol):
    mode: str

    def now(self) -> float: ...
    def charge(self, seconds: float) -> None: ...
    def sleep(self, seconds: float) -> None: ...


class WallClock:
    """Real (monotonic) time, measured from ``origin`` (episode start) so that tick boundaries are
    at integer multiples of ``tick_s`` — the runner's ticker is scheduled on the same origin."""
    mode = "wall"

    def __init__(self, origin: Optional[float] = None) -> None:
        self.origin = time.monotonic() if origin is None else origin

    def now(self) -> float:
        return time.monotonic() - self.origin

    def charge(self, seconds: float) -> None:  # real latency already elapsed
        pass

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class _Task:
    """One ``ctx.spawn`` task as the virtual clock sees it."""
    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state = "queued"      # queued | busy | parked | detached | done


class VirtualClock:
    """Deterministic discrete-event clock.

    Only the *driver* thread (the harness's ``run_episode`` thread) moves time forward: the model
    latency and sleeps it is charged advance the clock, and ``on_advance(t0, t1)`` lets the runner
    apply the game ticks inside the interval. A ``ctx.spawn`` worker charged for a background call
    parks until the clock reaches its wake time. The driver advances in segments: at each worker's
    wake it releases that worker and waits — at most ``handoff_s`` of real time — until every
    worker has parked again or finished, so a 2 s background Astra call overlaps exactly 2 s of the
    driver's Jev calls and its result is visible at that virtual instant, independent of thread
    scheduling. While the driver waits on a spawned future (``Future.result()``) the parked workers
    move time themselves, earliest wake first; a driver that is idle for ``idle_s`` of real time
    without telling the clock (``cf.wait``, ``time.sleep``, a polling loop) is treated the same way
    and the fallback is reported through ``on_event``. A worker still busy after ``handoff_s`` (a
    slow real backend) is detached: time no longer waits for it until it parks again.
    """
    mode = "virtual"

    def __init__(self, idle_s: float = 1.0, handoff_s: float = 1.0) -> None:
        self._t = 0.0
        self.on_advance: Optional[Callable[[float, float], None]] = None
        self.on_event: Optional[Callable[[str, dict[str, Any]], None]] = None
        self._cv = threading.Condition()
        self._driver: Optional[int] = None
        self._parked: dict[int, tuple[float, Optional[_Task]]] = {}
        self._released: set[int] = set()
        self._ids = itertools.count(1)
        self._tasks: set[_Task] = set()
        self._tls = threading.local()
        self._driver_charging = False
        self._driver_blocked = 0
        self._driver_in_call = 0
        self._advancing = False
        self._closed = False
        self._last_driver = time.monotonic()
        self.idle_s = idle_s
        self.handoff_s = handoff_s

    def set_driver(self, ident: Optional[int] = None) -> None:
        self._driver = ident if ident is not None else threading.get_ident()

    def now(self) -> float:
        return self._t

    def _is_driver(self) -> bool:
        return self._driver is None or threading.get_ident() == self._driver

    def _busy(self) -> int:
        return sum(t.state in ("queued", "busy") for t in self._tasks)

    def _running(self) -> int:
        return sum(t.state == "busy" for t in self._tasks)

    def _report(self, event: str, **data: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, {"clock": self._t, **data})
            except Exception:  # noqa: BLE001 — reporting must never break the clock
                pass

    # -- time ------------------------------------------------------------
    def _advance_to(self, t1: float) -> None:
        # called with the condition held; one thread applies game ticks at a time
        while self._advancing:
            self._cv.wait()
        t0 = self._t
        if t1 <= t0:
            return
        self._t = t1
        self._advancing = True
        cb = self.on_advance
        try:
            if cb is not None:
                self._cv.release()
                try:
                    cb(t0, t1)
                finally:
                    self._cv.acquire()
        finally:
            self._advancing = False
            self._cv.notify_all()

    def _release_due(self) -> None:
        for wid, (wake, task) in list(self._parked.items()):
            if wake <= self._t:
                del self._parked[wid]
                self._released.add(wid)
                if task is not None:
                    task.state = "busy"
        self._cv.notify_all()

    def _settle(self) -> None:
        """Wait until every spawned worker has parked or finished (bounded by handoff_s)."""
        deadline = time.monotonic() + self.handoff_s
        while self._busy() and not self._closed:
            left = deadline - time.monotonic()
            if left <= 0:
                stuck = [t for t in self._tasks if t.state in ("queued", "busy")]
                for t in stuck:
                    t.state = "detached"
                self._report("virtual_handoff_timeout", workers=len(stuck), handoff_s=self.handoff_s)
                return
            self._cv.wait(left)

    def charge(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        with self._cv:
            if self._is_driver():
                self._driver_charge(seconds)
            else:
                self._worker_charge(seconds)

    def _driver_charge(self, seconds: float) -> None:
        target = self._t + seconds
        self._driver_charging = True
        try:
            while not self._closed:
                self._settle()
                due = min((w for w, _ in self._parked.values() if w <= target), default=None)
                if due is None:
                    self._advance_to(target)
                    break
                self._advance_to(due)
                self._release_due()
        finally:
            self._driver_charging = False
            self._last_driver = time.monotonic()
            self._cv.notify_all()

    def _worker_charge(self, seconds: float) -> None:
        if self._closed:
            raise EpisodeOver("episode is over")
        wake = self._t + seconds
        if wake <= self._t:
            return
        wid = next(self._ids)
        task: Optional[_Task] = getattr(self._tls, "task", None)
        self._parked[wid] = (wake, task)
        if task is not None:
            task.state = "parked"
        self._cv.notify_all()                      # a settling driver may be waiting for us
        try:
            while wid not in self._released:
                if self._closed:
                    raise EpisodeOver("episode is over")
                self._cv.wait(min(0.05, self.idle_s))
                if wid not in self._released:
                    self._advance_for_waiting_driver()
        finally:
            self._parked.pop(wid, None)
            self._released.discard(wid)

    def _advance_for_waiting_driver(self) -> None:
        # called by a parked worker with the condition held
        # queued tasks are ignored here: with every pool thread parked they could never start
        if self._advancing or self._driver_charging or self._running() or not self._parked:
            return
        blocked = self._driver_blocked > 0
        idle = (not self._driver_in_call and time.monotonic() - self._last_driver >= self.idle_s)
        if not (blocked or idle):
            return
        wake = min(w for w, _ in self._parked.values())
        t0 = self._t
        self._advance_to(wake)
        self._release_due()
        if not blocked:
            self._last_driver = time.monotonic()
            self._report("virtual_idle_advance", t0=t0, t1=wake, idle_s=self.idle_s)

    def sleep(self, seconds: float) -> None:
        self.charge(seconds)

    # -- what the context tells the clock --------------------------------
    def task_submitted(self) -> _Task:
        with self._cv:
            task = _Task()
            self._tasks.add(task)
            return task

    def task_started(self, task: _Task) -> None:
        self._tls.task = task
        with self._cv:
            if task.state == "queued":
                task.state = "busy"
            self._cv.notify_all()

    def task_finished(self, task: _Task) -> None:
        self._tls.task = None
        with self._cv:
            task.state = "done"
            self._tasks.discard(task)
            self._cv.notify_all()

    @contextlib.contextmanager
    def driver_blocked(self) -> Iterator[None]:
        """The driver waits on a spawned future: parked workers may move time."""
        if self._driver is None or threading.get_ident() != self._driver:
            yield
            return
        with self._cv:
            self._driver_blocked += 1
            self._cv.notify_all()
        try:
            yield
        finally:
            with self._cv:
                self._driver_blocked -= 1
                self._last_driver = time.monotonic()

    @contextlib.contextmanager
    def driver_call(self) -> Iterator[None]:
        """The driver waits on a model backend: it is busy, not idle."""
        if self._driver is None or threading.get_ident() != self._driver:
            yield
            return
        with self._cv:
            self._driver_in_call += 1
        try:
            yield
        finally:
            with self._cv:
                self._driver_in_call -= 1
                self._last_driver = time.monotonic()

    def close(self) -> None:
        """Episode over: parked workers raise EpisodeOver, driver charges stop moving time."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class _SpawnFuture(cf.Future):
    """Future of a ``ctx.spawn`` task. Waiting on it from the episode's driver thread tells the
    virtual clock the driver is blocked, so background work can move time forward."""

    def __init__(self, clock: Any) -> None:
        super().__init__()
        self._clock = clock

    def _blocked(self):
        return self._clock.driver_blocked() if isinstance(self._clock, VirtualClock) else contextlib.nullcontext()

    def result(self, timeout: Optional[float] = None) -> Any:
        with self._blocked():
            return super().result(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        with self._blocked():
            return super().exception(timeout)


class _Call:
    """An in-flight model call; ``reason`` is set when the evaluator cancels it."""
    __slots__ = ("reason",)

    def __init__(self) -> None:
        self.reason: Optional[str] = None


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
        ticks_fn: Optional[Callable[[], int]] = None,
        tick_cv: Optional[threading.Condition] = None,
    ) -> None:
        self.tick_s = tick_s
        self._ticks = ticks_fn
        self._tick_cv = tick_cv
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
        self._pool = cf.ThreadPoolExecutor(max_workers=16, thread_name_prefix="harness")
        self._lock = threading.Lock()
        self._inflight: set[_Call] = set()
        self.last_system2: Optional[System2Result] = None
        self.errors: list[str] = []
        self.unavailable: Optional[ModelUnavailable] = None
        self._closed = threading.Event()
        self._futures: list[tuple[_SpawnFuture, cf.Future]] = []
        # Decision provenance (reported, never used for selection): how many posted
        # actions were preceded by a fresh System-1 / System-2 answer since the last act.
        self.acts = 0
        self.acts_after_s1 = 0
        self.acts_after_s2 = 0
        self._s1_credit = False
        self._s2_credit = False

    # -- game -----------------------------------------------------------
    def act(self, action: Any) -> Any:
        """Realtime: post the current action (instant). Turn: step and return obs."""
        if self.done:
            return self._observe()
        self.budget.check()
        out = self._act(action)
        self.acts += 1
        if self._s1_credit:
            self.acts_after_s1 += 1
            self._s1_credit = False
        if self._s2_credit:
            self.acts_after_s2 += 1
            self._s2_credit = False
        if self.game_kind == "realtime" and self.clock.mode == "virtual":
            self.clock.charge(self.act_overhead_s)
        return out

    def wait_for_tick(self) -> None:
        """Block until the game has applied its next tick (realtime games only)."""
        if self.done or self.game_kind != "realtime" or not self.tick_s:
            return
        self.budget.check()
        if self.clock.mode == "wall" and self._tick_cv is not None and self._ticks is not None:
            # Wait on the tick counter, not the clock: right after a boundary whose tick is still
            # pending, sleeping to the next boundary would skip a whole decision window.
            target = self._ticks() + 1
            with self._tick_cv:
                self._tick_cv.wait_for(lambda: self._ticks() >= target or self.done,
                                       timeout=2 * self.tick_s + 0.05)
            return
        now = self.clock.now()
        nxt = (int(now / self.tick_s + 1e-9) + 1) * self.tick_s
        self.clock.sleep(max(1e-6, nxt - now))

    def observe(self) -> Any:
        return self._observe()

    @property
    def done(self) -> bool:
        return self._closed.is_set() or self._done()

    # -- models ---------------------------------------------------------
    @contextlib.contextmanager
    def _call(self) -> Iterator[_Call]:
        call = _Call()
        with self._lock:
            self._inflight.add(call)
        driver_call = self.clock.driver_call() if isinstance(self.clock, VirtualClock) else contextlib.nullcontext()
        try:
            with cancel_scope(lambda: call.reason), driver_call:
                yield call
        finally:
            with self._lock:
                self._inflight.discard(call)

    def cancel_inflight(self, reason: str) -> int:
        """Cancel the model calls in flight right now (calls started later are unaffected)."""
        with self._lock:
            calls = list(self._inflight)
        for c in calls:
            c.reason = c.reason or reason
        return len(calls)

    def _charge_since(self, started: float, seconds: float) -> None:
        if self.clock.mode == "virtual":
            self.clock.charge(max(0.0, started + seconds - self.clock.now()))

    def _model_call(self, kind: str, started: float, timeout_s: float, fn: Callable[[], Any]) -> Any:
        t0 = time.monotonic()
        with self._call() as call:
            try:
                out = fn()
            except ModelUnavailable as exc:
                self.unavailable = exc
                raise
            except ModelCallCancelled as exc:
                raise EpisodeOver(f"{kind} call cancelled: {call.reason or exc}") from None
            except Exception as exc:
                if call.reason or self._closed.is_set():
                    raise EpisodeOver(f"{kind} call ended with the episode: {call.reason or exc}") from None
                # a failed call takes time too (a timeout costs its timeout) — never free in virtual mode
                self._charge_since(started, timeout_s if isinstance(exc, TimeoutError) else time.monotonic() - t0)
                raise
        if self._closed.is_set():
            raise EpisodeOver(f"{kind} answer arrived after the episode closed")
        return out

    def think(self, prompt: str, *, session_id: Optional[str] = None, system: Optional[str] = None,
              images: Optional[list[str]] = None, timeout_s: float = 300.0) -> System2Result:
        if self.system2 is None:
            raise RuntimeError("no System-2 model in this task")
        self._check_open()
        self.budget.reserve("system2")
        timeout_s = min(timeout_s, self.budget.remaining()["wall_s"])
        started = self.clock.now()
        self.trace.write({"type": "model_request", "kind": "system2", "clock": started, "prompt": prompt,
                          "session_id": session_id, "timeout_s": timeout_s,
                          **({"images": [str(i) for i in images]} if images else {})})
        kw = {"images": images} if images else {}
        res = self._model_call("system2", started, timeout_s, lambda: self.system2.run(
            prompt, session_id=session_id, system=system, agentic=False, timeout_s=timeout_s,
            trace=self.trace, **kw))
        self._charge_since(started, self._latency_override.get("system2", res.latency_s))
        self.last_system2 = res
        self._s2_credit = True
        return res

    def decide(self, state: Any, questions: Mapping[str, Any], *, timeout_s: float = 5.0) -> dict[str, System1Answer]:
        if self.system1 is None:
            raise RuntimeError("no System-1 model in this task")
        self._check_open()
        self.budget.reserve("system1")
        timeout_s = min(timeout_s, self.budget.remaining()["wall_s"])
        started = self.clock.now()
        self.trace.write({"type": "model_request", "kind": "system1", "clock": started, "timeout_s": timeout_s,
                          "state": state, "questions": {
                              key: value.to_json() if hasattr(value, "to_json") else value
                              for key, value in questions.items()}})
        t0 = time.monotonic()
        ans = self._model_call("system1", started, timeout_s, lambda: self.system1.decide(
            state, questions, timeout_s=timeout_s, trace=self.trace))
        lat = time.monotonic() - t0
        self._charge_since(started, self._latency_override.get("system1", getattr(self.system1, "latency_s", lat)))
        self._s1_credit = True
        return ans

    def _check_open(self) -> None:
        if self._closed.is_set():
            raise EpisodeOver("episode is closed")

    def spawn(self, fn: Callable[[], Any]) -> cf.Future:
        self._check_open()
        fut = _SpawnFuture(self.clock)
        vclock = self.clock if isinstance(self.clock, VirtualClock) else None
        task = vclock.task_submitted() if vclock is not None else None

        def run() -> None:
            if task is not None:
                vclock.task_started(task)
            try:
                if not fut.set_running_or_notify_cancel():
                    return
                try:
                    result = fn()
                except BaseException as e:  # noqa: BLE001 — delivered through the future
                    fut.set_exception(e)
                else:
                    fut.set_result(result)
            finally:
                if task is not None:
                    vclock.task_finished(task)

        try:
            inner = self._pool.submit(run)
        except RuntimeError:             # pool already shut down: the episode is over
            if task is not None:
                vclock.task_finished(task)
            raise EpisodeOver("episode is closed") from None
        self._futures.append((fut, inner))
        return fut

    def log(self, *args: Any, **event: Any) -> None:
        """Free-form harness event into the trace. ``ctx.log(event="x", k=v)``;
        a leading positional str becomes ``event``, a leading dict is merged."""
        for a in args:
            if isinstance(a, Mapping):
                event = {**a, **event}
            elif isinstance(a, str) and "event" not in event:
                event["event"] = a
            else:
                event.setdefault("args", []).append(a)
        event.setdefault("type", "harness")
        event["clock"] = self.clock.now()
        self.trace.write(event)

    def begin_close(self, reason: str = "episode closed") -> None:
        """Refuse new calls, cancel in-flight model calls, release virtual-clock waiters."""
        self._closed.set()
        self.cancel_inflight(reason)
        if isinstance(self.clock, VirtualClock):
            self.clock.close()

    def close(self, grace_s: float = 0.0) -> int:
        """End the episode and wait up to ``grace_s`` for spawned work; returns how many tasks
        are still running (an episode error)."""
        self.begin_close()
        futs = [f for f, _ in self._futures]
        if grace_s > 0 and futs:
            cf.wait(futs, timeout=grace_s)
        self._pool.shutdown(wait=False, cancel_futures=True)
        for fut, inner in self._futures:
            if inner.cancelled() and not fut.done():
                fut.cancel()
        return sum(not f.done() for f, _ in self._futures)


class HarnessProgram(Protocol):
    def run_episode(self, game: Any) -> None: ...


_LOADS = itertools.count(1)


def load_harness_module(path: Path | str, name: Optional[str] = None):
    """Import a harness file as a fresh module (each candidate isolated)."""
    path = Path(path)
    mod_name = name or f"_harness_{abs(hash(str(path.resolve())))}_{next(_LOADS)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load harness at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    if not hasattr(module, "build"):
        sys.modules.pop(mod_name, None)
        raise ImportError(f"harness {path} defines no build(ctx)")
    return module


def unload_harness_module(module: Any) -> None:
    """Drop a harness module from ``sys.modules`` once its episode is over (no growth over a search)."""
    name = getattr(module, "__name__", None)
    if name and sys.modules.get(name) is module:
        del sys.modules[name]


__all__ = ["Budget", "BudgetExceeded", "EpisodeOver", "Clock", "WallClock", "VirtualClock", "HarnessContext",
           "HarnessProgram", "load_harness_module", "unload_harness_module", "ModelUnavailable"]
