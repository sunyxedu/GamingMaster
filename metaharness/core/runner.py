"""Evaluate one harness candidate on one game over several seeds.

Produces per-episode scores plus the *latency accounting* that makes tier-2
harness comparisons honest: how many ticks ran on a stale action, model
latency distribution, and how many System-1/2 calls were spent.

Episode lifecycle (``run_episode``): the game is reset, then the per-episode
wall-clock deadline starts (``Budget.start``) — one deadline, the smaller of the
wall budget and the harness timeout, which is also what ``ctx.budget`` reports.
The harness runs on its own thread. At the deadline the episode is closed (ctx
calls refused, in-flight model calls cancelled) and the harness gets
``grace_s`` to return. An episode that ends by the game, by returning, by a
call budget or by the deadline keeps its score; only harness exceptions, a
harness that does not return within the grace period, background work still
running after it, and game failures are *errors* (``ended_by`` records which
way it ended). Every access to the game object is serialised by one lock, so
generated adaptors need not be thread-safe against the ticker, the harness and
``ctx.spawn`` workers.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import statistics
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..games.base import close_game
from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from .contract import (Budget, BudgetExceeded, HarnessContext, VirtualClock, WallClock, load_harness_module,
                       unload_harness_module)

PROTOCOL_VERSION = 3          # 3: one deadline, budget/deadline ends keep the score, runner-counted steps
DEFAULT_GRACE_S = 10.0        # after the deadline (and for spawned work after run_episode returns)
STEP_OBS_CAP = 16_000         # characters of an observation kept per step record
FINAL_OBS_CAP = 64_000        # ... at episode start / end in the trace
EVAL_OBS_CAP = 4_000          # ... of the final observation in eval.json


@dataclass
class EpisodeResult:
    seed: int
    score: float
    steps: int
    system1_calls: int
    system2_calls: int
    stale_ticks: int = 0          # realtime: ticks applied without a fresh decision
    decided_ticks: int = 0
    latency_s1: list[float] = field(default_factory=list)
    latency_s2: list[float] = field(default_factory=list)
    wall_s: float = 0.0
    error: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)
    acts: int = 0                 # actions posted by the harness
    acts_after_s1: int = 0        # ... preceded by a fresh System-1 answer
    acts_after_s2: int = 0        # ... preceded by a fresh System-2 answer
    failed_s1: int = 0            # model calls that errored / timed out (not counted in latency)
    failed_s2: int = 0
    timeouts_s1: int = 0          # ... of which ran into their timeout
    timeouts_s2: int = 0
    ended_by: str = ""            # game_done | harness_return | call_budget | deadline | virtual_cap |
                                  # harness_error | game_error | reset_failed | construction_failed
    final_observation: Optional[str] = None     # JSON text, capped
    invalid: Optional[str] = None               # the adaptor reported an invalid trial (infrastructure)
    tokens_s2: list[Optional[int]] = field(default_factory=list)   # output tokens per successful S2 call
    backends: dict[str, list[str]] = field(default_factory=dict)   # model ids that answered, per kind

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_s1"] = _stats(self.latency_s1)
        d["latency_s2"] = _stats(self.latency_s2)
        d["calls_s2"] = [[round(lat, 3), tok] for lat, tok in zip(self.latency_s2, self.tokens_s2)]
        del d["tokens_s2"]
        return d


def _frac(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": xs[len(xs) // 2],
            "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}


def _tb(e: BaseException, limit: int) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))[-limit:]


def _last_line(text: str, limit: int = 200) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return (lines[-1] if lines else "")[:limit]


def _obs_text(obs: Any, cap: int) -> str:
    try:
        s = json.dumps(obs, default=str, separators=(",", ":"))
    except Exception as e:  # noqa: BLE001
        s = f"<observation not serialisable: {type(e).__name__}>"
    return s if len(s) <= cap else s[:cap] + "…(truncated)"


def _obs_field(obs: Any, cap: int) -> Any:
    """An observation for a trace record: the value itself when small, else truncated JSON text."""
    s = _obs_text(obs, cap)
    if s.endswith("…(truncated)") or s.startswith("<observation"):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return s


@dataclass
class EvalResult:
    harness_path: str
    game: str
    episodes: list[EpisodeResult]
    fatal: Optional[str] = None       # harness failed to load / model unavailable
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_score(self) -> Optional[float]:
        ok = [e.score for e in self.episodes]
        return statistics.fmean(ok) if ok else None

    @property
    def stale_fraction(self) -> Optional[float]:
        tot = sum(e.stale_ticks + e.decided_ticks for e in self.episodes)
        return (sum(e.stale_ticks for e in self.episodes) / tot) if tot else None

    def to_json(self) -> dict[str, Any]:
        return {
            "harness_path": self.harness_path, "game": self.game, "fatal": self.fatal,
            "mean_score": self.mean_score, "stale_fraction": self.stale_fraction,
            "n_episodes": len(self.episodes),
            "system1_calls": sum(e.system1_calls for e in self.episodes),
            "system2_calls": sum(e.system2_calls for e in self.episodes),
            "mean_steps": (statistics.fmean(e.steps for e in self.episodes) if self.episodes else None),
            "failed_s1": sum(e.failed_s1 for e in self.episodes),
            "failed_s2": sum(e.failed_s2 for e in self.episodes),
            "timeouts_s1": sum(e.timeouts_s1 for e in self.episodes),
            "timeouts_s2": sum(e.timeouts_s2 for e in self.episodes),
            "ended_by": dict(collections.Counter(e.ended_by for e in self.episodes)),
            "s1_informed_fraction": _frac(sum(e.acts_after_s1 for e in self.episodes), sum(e.acts for e in self.episodes)),
            "s2_informed_fraction": _frac(sum(e.acts_after_s2 for e in self.episodes), sum(e.acts for e in self.episodes)),
            "errors": [e.error for e in self.episodes if e.error],
            "episodes": [e.summary() for e in self.episodes],
            "metadata": self.metadata,
            "score_std": (statistics.stdev(e.score for e in self.episodes)
                          if len(self.episodes) > 1 else None),
        }


class _GameView:
    """What the harness sees: observations + metadata, never the stepping API.

    Every access goes through ``lock`` (shared with the ticker / stepping code), so the
    harness, ``ctx.spawn`` workers and the wall ticker never overlap inside the game object."""

    def __init__(self, game: Any, stopped: threading.Event, lock: Optional[threading.RLock] = None) -> None:
        self._g = game
        self._stopped = stopped
        self._lock = lock or threading.RLock()
        self.name = game.name

    @property
    def action_spec(self) -> Any:      # live: an adaptor may change its legal actions during play
        with self._lock:
            return self._g.action_spec

    @property
    def tick_s(self) -> Optional[float]:
        with self._lock:
            return getattr(self._g, "tick_s", None)

    def observe(self) -> Any:
        with self._lock:
            return self._g.observe()

    def describe(self) -> str:
        with self._lock:
            return self._g.describe()

    def game_done(self) -> bool:
        with self._lock:
            return bool(self._g.done)

    @property
    def done(self) -> bool:
        return self._stopped.is_set() or self.game_done()

    @property
    def score(self) -> float:
        with self._lock:
            return self._g.score

    @property
    def ticks(self) -> int:
        with self._lock:
            return getattr(self._g, "ticks", 0)

    def info(self) -> dict[str, Any]:
        # Game.info() is evaluator diagnostics and can contain hidden rules.
        return {"score": self.score, "ticks": self.ticks, "game_done": self.game_done()}


class _Mailbox:
    def __init__(self, noop: Any) -> None:
        self._lock = threading.Lock()
        self._action = noop
        self._fresh = False
        self.noop = noop

    def post(self, action: Any) -> None:
        with self._lock:
            self._action, self._fresh = action, True

    def take(self) -> tuple[Any, bool]:
        with self._lock:
            a, f = self._action, self._fresh
            self._fresh = False
            return a, f


_TIMEOUT_WORDS = ("timeout", "timed out", "deadline", "exceeded")


def run_episode(
    module: Any,
    game: Any,
    *,
    seed: int,
    system2: Optional[System2Model],
    system1: Optional[System1Model],
    trace: TraceSink,
    memory: dict[str, Any],
    budget: Budget,
    clock_mode: str = "wall",
    latency_override: Optional[dict[str, float]] = None,
    harness_timeout_s: float = 600.0,
    grace_s: float = DEFAULT_GRACE_S,
    max_virtual_s: Optional[float] = None,
) -> EpisodeResult:
    realtime = hasattr(game, "tick")
    lock = threading.RLock()
    res = EpisodeResult(seed=seed, score=0.0, steps=0, system1_calls=0, system2_calls=0)
    try:
        with lock:
            game.reset(seed)
    except ModelUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 — a scenario the adaptor cannot provide is an episode error, not a crash
        res.error = "game reset failed: " + _tb(e, 2000)
        res.ended_by = "reset_failed"
        trace.write({"type": "episode_end", **res.summary()})
        return res
    budget.start()
    deadline_s = max(0.0, min(harness_timeout_s, budget.max_wall_s))
    budget.max_wall_s = deadline_s           # ctx.budget reports the deadline that is enforced
    stop = threading.Event()
    view = _GameView(game, stop, lock)
    tick_cv = threading.Condition()
    accepting_trace = threading.Event()
    accepting_trace.set()
    t_wall0 = time.monotonic()
    s1_lat: list[float] = []
    s2_lat: list[float] = []
    s2_tok: list[Optional[int]] = []
    backends: dict[str, set[str]] = {"system1": set(), "system2": set()}
    game_errors: list[str] = []
    capped = threading.Event()
    holder: dict[str, HarnessContext] = {}
    cancelled_calls = [0]
    # tick count / end flag published by the ticker under tick_cv, so waiters never need the game
    # lock while holding tick_cv (a hung tick() must not deadlock the evaluator)
    published = {"ticks": 0, "over": False, "score": 0.0}

    # tracing hooks for latency (trace records carry latency_s)
    class _Tee(TraceSink):
        def __init__(self, inner: TraceSink) -> None:
            self.inner = inner
            self.records = inner.records
            self.path = inner.path

        def write(self, record):
            if not accepting_trace.is_set():
                return
            r = dict(record)
            r["seed"] = seed
            if r.get("type") == "model_call":
                kind = "system1" if r.get("kind") == "system1" else "system2"
                if r.get("ok"):
                    if kind == "system1":
                        s1_lat.append(float(r.get("latency_s", 0.0)))
                    else:
                        s2_lat.append(float(r.get("latency_s", 0.0)))
                        tok = (r.get("usage") or {}).get("output_tokens")
                        s2_tok.append(int(tok) if isinstance(tok, (int, float)) and not isinstance(tok, bool) else None)
                    if r.get("backend"):
                        backends[kind].add(str(r["backend"]))
                else:
                    error = str(r.get("error") or "").lower()
                    if "cancelled" in error:
                        cancelled_calls[0] += 1        # ended with the episode: not a backend failure
                    elif kind == "system1":
                        res.failed_s1 += 1
                        res.timeouts_s1 += any(w in error for w in _TIMEOUT_WORDS)
                    else:
                        res.failed_s2 += 1
                        res.timeouts_s2 += any(w in error for w in _TIMEOUT_WORDS)
            self.inner.write(r)

    tee = _Tee(trace)
    stale = decided = steps = 0
    ticker: Optional[threading.Thread] = None
    mailbox = _Mailbox(game.action_spec.noop if realtime else None)

    def _wake_waiters() -> None:
        with tick_cv:
            tick_cv.notify_all()

    def _game_failed(what: str, e: BaseException) -> None:
        msg = f"game {what} failed: " + _tb(e, 2000)
        if not game_errors:
            game_errors.append(msg)
        tee.write({"type": "game_error", "error": msg})
        stop.set()
        if "ctx" in holder:
            holder["ctx"].cancel_inflight("game failed")
        _wake_waiters()

    def _game_over() -> None:
        if "ctx" in holder:
            holder["ctx"].cancel_inflight("game over")    # answers about a finished game are moot
        _wake_waiters()

    def _trace_observation(**where: Any) -> None:
        try:
            tee.write({"type": "observation", **where, "obs": _obs_field(view.observe(), STEP_OBS_CAP)})
        except Exception as e:  # noqa: BLE001
            tee.write({"type": "observation", **where, "error": f"{type(e).__name__}: {e}"[:300]})

    if realtime:
        tick_s = float(game.tick_s)
        last_score: list[Any] = [object()]

        def _apply_tick() -> bool:
            nonlocal stale, decided
            a, fresh = mailbox.take()
            try:
                with lock:
                    game.tick(a)
                    n, score, over = game.ticks, game.score, bool(game.done)
            except Exception as e:  # noqa: BLE001 — a failing game must end the episode, not freeze it
                _game_failed(f"tick (action {a!r})", e)
                return False
            if fresh:
                decided += 1
            else:
                stale += 1
            with tick_cv:
                published.update(ticks=n, over=over, score=score)
                tick_cv.notify_all()
            tee.write({"type": "tick", "tick": n, "action": a, "fresh": fresh, "score": score})
            if score != last_score[0] or over:
                last_score[0] = score
                _trace_observation(tick=n)
            if over:
                _game_over()
            return not over

        if clock_mode == "virtual":
            clock = VirtualClock()

            def _advance(t0: float, t1: float) -> None:
                if max_virtual_s is not None and t1 >= max_virtual_s and not stop.is_set():
                    capped.set()
                    stop.set()
                    _wake_waiters()
                target = int(t1 / tick_s + 1e-9)
                while view.ticks < target and not stop.is_set() and not view.game_done():
                    if not _apply_tick():
                        break
            clock.on_advance = _advance
        else:
            clock = WallClock()

            def _ticker() -> None:
                # phase-aligned: tick n fires at origin + n * tick_s
                n = 1
                while not stop.is_set() and not view.game_done():
                    delay = clock.origin + n * tick_s - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)
                    if stop.is_set():
                        break
                    if not _apply_tick():
                        break
                    n += 1
                _wake_waiters()
            ticker = threading.Thread(target=_ticker, name="game-ticker", daemon=True)

        act_fn = mailbox.post
    else:
        clock = VirtualClock() if clock_mode == "virtual" else WallClock()

        def act_fn(action: Any) -> Any:
            nonlocal steps
            if stop.is_set():
                return view.observe()
            try:
                with lock:
                    obs = game.step(action)
            except Exception as e:
                # an illegal action is the harness's to handle: the exception reaches it
                tee.write({"type": "step_error", "action": action, "error": f"{type(e).__name__}: {e}"[:500]})
                raise
            steps += 1
            try:
                with lock:
                    score, over = game.score, bool(game.done)
            except Exception as e:  # noqa: BLE001 — the game, not the harness, is broken
                _game_failed("state read after a step", e)
                return obs
            published.update(ticks=steps, over=over, score=score)
            tee.write({"type": "step", "step": steps, "action": action, "score": score,
                       "obs": _obs_field(obs, STEP_OBS_CAP)})
            if over:
                _game_over()
            return obs

    ctx = HarnessContext(system2=system2, system1=system1, trace=tee, clock=clock, budget=budget,
                         memory=memory, game_kind="realtime" if realtime else "turn",
                         act_fn=act_fn, observe_fn=view.observe, done_fn=lambda: view.done,
                         latency_override=latency_override,
                         tick_s=tick_s if realtime else None,
                         ticks_fn=(lambda: published["ticks"]) if realtime else None,
                         tick_cv=tick_cv if realtime else None,
                         tick_over_fn=lambda: published["over"] or stop.is_set(),
                         on_unavailable=lambda: (stop.set(), _wake_waiters()))
    holder["ctx"] = ctx
    if isinstance(clock, VirtualClock):
        clock.on_event = lambda event, data: tee.write({"type": event, **data})
    try:
        start_obs = _obs_field(view.observe(), FINAL_OBS_CAP)
    except Exception as e:  # noqa: BLE001
        start_obs = f"<observe() failed: {type(e).__name__}: {e}>"[:300]
    tee.write({"type": "episode_start", "game": game.name, "realtime": realtime, "clock": clock_mode,
               "deadline_s": deadline_s, "obs": start_obs})
    err: Optional[str] = None
    ended_by = ""
    done_flag = threading.Event()
    exc: list[BaseException] = []

    def _run() -> None:
        if isinstance(clock, VirtualClock):
            clock.set_driver()
        try:
            program = module.build(ctx)
            program.run_episode(view)
        except BaseException as e:  # noqa: BLE001
            exc.append(e)
        finally:
            done_flag.set()

    th = threading.Thread(target=_run, name="harness", daemon=True)
    pending = 0
    try:
        if ticker:
            clock.origin = time.monotonic()
            ticker.start()
        th.start()
        finished = done_flag.wait(max(0.0, deadline_s - budget.elapsed()))
        if not finished:
            ended_by = "deadline"
            stop.set()
            ctx.begin_close("episode deadline")         # first: a call ending now must see the close
            _wake_waiters()
            tee.write({"type": "deadline", "deadline_s": deadline_s, "grace_s": grace_s})
            finished = done_flag.wait(grace_s)
            if not finished:
                err = (f"harness did not return within {grace_s:.1f}s after the episode deadline "
                       f"({deadline_s:.1f}s); ctx calls were refused and in-flight model calls cancelled")
        if finished and exc:
            e = exc[0]
            if isinstance(e, ModelUnavailable):
                raise e
            if isinstance(e, BudgetExceeded):
                ended_by = ended_by or ("call_budget" if e.kind in ("system1", "system2") else
                                        "deadline" if e.kind == "wall" else "")
            else:
                err = _tb(e, 3000)
                ended_by = ended_by or "harness_error"
    finally:
        stop.set()
        _wake_waiters()
        if ticker and ticker.is_alive():
            ticker.join(timeout=2.0)
        pending = ctx.close(grace_s=grace_s if done_flag.is_set() else 0.0)
        accepting_trace.clear()
    if ctx.unavailable is not None:
        raise ctx.unavailable
    if pending and err is None:
        err = (f"{pending} ctx.spawn task(s) still running {grace_s:.0f}s after run_episode returned "
               "(model calls are cancelled or end with their own timeout; other background work must "
               "stop when ctx.done)")
    if game_errors:
        err = err or game_errors[0]
        ended_by = "game_error"
    game_done = False
    # The game lock is free unless a call into the game never returned (a hung server): never wait
    # for it unboundedly — record the hang and use what the ticker / steps published.
    breakdown: Optional[dict[str, float]] = None
    breakdown_error: Optional[str] = None
    hung = not lock.acquire(timeout=2.0)          # a game call returns in well under that
    if hung:
        err = ("game hung: a call into the game (tick/step/observe) did not return, so neither the "
               "harness nor the evaluator could use it" + (f" — also: {_last_line(err)}" if err else ""))
        ended_by = "game_error"
        res.score = float(published["score"]) if math.isfinite(float(published["score"])) else 0.0
        res.steps = int(published["ticks"])
        final_obs, game_done = "<game hung>", bool(published["over"])
    else:
        try:
            try:
                score = float(game.score)
                game_done = bool(game.done)
                res.steps = int(getattr(game, "ticks", 0)) if realtime else steps
                if not math.isfinite(score):
                    raise ValueError(f"the game reports a non-finite score {score!r}")
                res.score = score
            except Exception as e:  # noqa: BLE001
                err = err or "game failed after the episode: " + _tb(e, 2000)
                res.steps = res.steps or steps
            try:
                final_obs = game.observe()
            except Exception as e:  # noqa: BLE001
                final_obs = f"<observe() failed: {type(e).__name__}: {e}>"[:300]
            try:
                info = game.info()
                # evaluator-only infrastructure signal; no other info() key is read (they can hold hidden rules)
                if isinstance(info, dict) and info.get("valid_trial") is False:
                    res.invalid = str(info.get("invalid_reason") or "the game reported an invalid trial")[:300]
            except Exception:  # noqa: BLE001 — diagnostics are optional
                pass
            # Optional adaptor hook: a numeric breakdown of the score (e.g. progress vs survival) so the
            # proposer can see *where* points came from instead of one scalar. Public, not hidden state.
            if hasattr(game, "score_breakdown"):
                try:
                    bd = game.score_breakdown()
                    if isinstance(bd, dict):
                        breakdown = {str(k): float(v) for k, v in bd.items() if isinstance(v, (int, float))}
                except Exception as e:  # noqa: BLE001
                    breakdown_error = str(e)[:200]
        finally:
            lock.release()
    if not ended_by:
        ended_by = ("game_done" if game_done else "virtual_cap" if capped.is_set() else
                    "harness_return" if done_flag.is_set() else "deadline")
    elif game_done and ended_by not in ("game_error", "harness_error"):
        ended_by = "game_done"
    res.system1_calls = budget.system1_calls
    res.system2_calls = budget.system2_calls
    res.stale_ticks, res.decided_ticks = stale, decided
    res.latency_s1, res.latency_s2, res.tokens_s2 = s1_lat, s2_lat, s2_tok
    res.wall_s = time.monotonic() - t_wall0
    res.error = err
    res.ended_by = ended_by
    res.final_observation = _obs_text(final_obs, EVAL_OBS_CAP)
    res.backends = {k: sorted(v) for k, v in backends.items() if v}
    res.info = {"score": res.score, "ticks": res.steps, "game_done": game_done}
    res.info.update({"pending_tasks": pending, "deadline_s": deadline_s, "harness_alive": th.is_alive(),
                     "cancelled_calls": cancelled_calls[0]})
    if breakdown is not None:
        res.info["score_breakdown"] = breakdown
    if breakdown_error is not None:
        res.info["score_breakdown_error"] = breakdown_error
    res.acts, res.acts_after_s1, res.acts_after_s2 = ctx.acts, ctx.acts_after_s1, ctx.acts_after_s2
    trace.write({"type": "episode_end", **res.summary(), "final_obs": _obs_field(final_obs, FINAL_OBS_CAP)})
    return res


def _copy_memory(memory: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """Deep copy of shared memory (a harness thread that outlived its deadline may still mutate it);
    a value that cannot be deep-copied (a lock, a generator) falls back to a shallow copy."""
    for _ in range(3):
        try:
            return copy.deepcopy(memory), None
        except RuntimeError:          # changed size during iteration
            time.sleep(0.01)
        except Exception as e:  # noqa: BLE001
            return dict(memory), f"shallow copy: a value cannot be deep-copied ({type(e).__name__}: {e})"[:300]
    raise RuntimeError("shared memory is still being modified by a previous episode's harness thread")


def evaluate_harness(
    harness_path: Path | str,
    game_factory: Any,
    seeds: list[int],
    *,
    system2: Optional[System2Model],
    system1: Optional[System1Model],
    out_dir: Optional[Path] = None,
    clock_mode: str = "wall",
    budget_per_episode: Optional[Budget] = None,
    latency_override: Optional[dict[str, float]] = None,
    harness_timeout_s: float = 600.0,
    game_kwargs: Optional[dict[str, Any]] = None,
    memory_mode: str = "episode",
    grace_s: float = DEFAULT_GRACE_S,
    game_name: Optional[str] = None,
    invalid_retries: int = 1,
    parallel: int = 1,
) -> EvalResult:
    """``parallel``: how many seeds play at once (each with its own game, module and trace). Shared
    memory orders the seeds, so it is always sequential. Episodes are reported in seed order."""
    if memory_mode not in {"episode", "shared"}:
        raise ValueError("memory_mode must be episode or shared")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation requires nonempty, unique seeds")
    if clock_mode not in {"wall", "virtual"}:
        raise ValueError("clock_mode must be wall or virtual")
    if latency_override and clock_mode != "virtual":
        raise ValueError("latency_override requires virtual clock")
    harness_path = Path(harness_path)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    result = EvalResult(harness_path=str(harness_path), game=game_name or "unknown", episodes=[])
    b = budget_per_episode or Budget()
    deadline_s = min(b.max_wall_s, harness_timeout_s)
    workers = max(1, int(parallel)) if memory_mode == "episode" else 1
    result.metadata = {
        "protocol_version": PROTOCOL_VERSION, "seeds": seeds, "expected_episodes": len(seeds),
        "game_kwargs": game_kwargs or {}, "clock_mode": clock_mode, "latency_override": latency_override,
        "memory_mode": memory_mode, "harness_timeout_s": harness_timeout_s, "deadline_s": deadline_s,
        "grace_s": grace_s, "parallel": min(workers, len(seeds)),
        "budget": {"max_system1_calls": b.max_system1_calls,
                   "max_system2_calls": b.max_system2_calls, "max_wall_s": b.max_wall_s},
        "models": {kind: None if model is None else {
            "name": model.name, "model": getattr(model, "model", None),
            "player_tools": getattr(model, "player_tools", None),
            "reasoning_effort": getattr(model, "reasoning_effort", None)}
            for kind, model in (("system1", system1), ("system2", system2))},
    }
    try:
        result.metadata["harness_sha256"] = hashlib.sha256(harness_path.read_bytes()).hexdigest()
        load_harness_module(harness_path)       # a load error is fatal before any episode
    except Exception as e:  # noqa: BLE001
        result.fatal = "load error: " + _tb(e, 3000)
        _dump(result, out_dir)
        return result
    meta_lock = threading.Lock()
    resolved: dict[str, set[str]] = {}
    stop = threading.Event()                 # a model outage: seeds not started yet are skipped

    def play(seed: int, memory: dict[str, Any], note: Optional[str]) -> Optional[EpisodeResult]:
        """One seed, replayed once on an invalid trial. Returns None on a fatal (recorded)."""
        for attempt in range(invalid_retries + 1):
            if stop.is_set():
                return None
            try:
                # Fresh game/module per episode: timed-out workers and globals never contaminate another seed.
                module = load_harness_module(harness_path)
            except Exception as e:  # noqa: BLE001 — the file changed under the evaluator
                with meta_lock:
                    result.fatal = result.fatal or "load error: " + _tb(e, 3000)
                stop.set()
                return None
            if note:
                with meta_lock:
                    result.metadata.setdefault("memory_resets", []).append({"seed": seed, "reason": note})
            trace_path = out_dir / f"trace_seed{seed}.jsonl" if out_dir else None
            trace = TraceSink(trace_path, append=False)
            budget = Budget(max_system2_calls=b.max_system2_calls, max_system1_calls=b.max_system1_calls,
                            max_wall_s=deadline_s)
            game = None
            ep: Optional[EpisodeResult] = None
            try:
                try:
                    game = game_factory(**(game_kwargs or {}))
                except ModelUnavailable:
                    raise
                except Exception as e:  # noqa: BLE001 — environment failure: an episode error, not a crash
                    ep = EpisodeResult(seed=seed, score=0.0, steps=0, system1_calls=0, system2_calls=0,
                                       error="game construction failed: " + _tb(e, 2000),
                                       ended_by="construction_failed")
                    trace.write({"type": "episode_end", **ep.summary()})
                else:
                    with meta_lock:
                        if result.game == "unknown":
                            result.game = game.name
                        result.metadata["game_kind"] = "realtime" if hasattr(game, "tick") else "turn"
                    ep = run_episode(module, game, seed=seed, system2=system2, system1=system1, trace=trace,
                                     memory=memory, budget=budget, clock_mode=clock_mode,
                                     latency_override=latency_override, harness_timeout_s=harness_timeout_s,
                                     grace_s=grace_s)
            except ModelUnavailable as e:
                with meta_lock:
                    result.fatal = result.fatal or f"model unavailable: {e}"
                stop.set()
                trace.write({"type": "episode_unavailable", "seed": seed, "error": str(e)})
            finally:
                trace.close()
                if game is not None:
                    close_err = close_game(game)
                    if close_err and ep is not None:
                        ep.info["close_error"] = close_err
                if ep is None or not ep.info.get("harness_alive"):
                    unload_harness_module(module)
            if ep is None or stop.is_set():
                return None
            if ep.invalid and attempt < invalid_retries:     # the failure may have raised as well
                # infrastructure failure the adaptor detected: replay the seed once on a fresh game
                if trace_path and trace_path.exists():
                    trace_path.rename(trace_path.with_name(f"trace_seed{seed}.invalid{attempt}.jsonl"))
                with meta_lock:
                    result.metadata.setdefault("invalid_trials", []).append({"seed": seed, "reason": ep.invalid})
                continue
            if ep.invalid:      # also when the failure raised: it is the infrastructure's, not the harness's
                ep.error = (f"invalid trial (reported by the game): {ep.invalid}"
                            + (f" — also: {_last_line(ep.error)}" if ep.error else ""))
            with meta_lock:
                for kind, names in ep.backends.items():
                    resolved.setdefault(kind, set()).update(names)
            return ep
        return None

    episodes: dict[int, EpisodeResult] = {}
    if workers == 1:
        carried: dict[str, Any] = {}           # shared mode: memory after the last accepted episode
        for seed in seeds:
            try:
                memory, note = _copy_memory(carried) if memory_mode == "shared" else ({}, None)
            except RuntimeError as e:   # the previous episode's harness never returned (already an error)
                memory, note = {}, str(e)
            ep = play(seed, memory, note)
            if ep is None:
                break
            carried = memory
            episodes[seed] = ep
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(seeds)), thread_name_prefix="eval-seed") as pool:
            futs = {seed: pool.submit(play, seed, {}, None) for seed in seeds}
            for seed, fut in futs.items():
                ep = fut.result()
                if ep is not None:
                    episodes[seed] = ep
    result.episodes = [episodes[s] for s in seeds if s in episodes]
    result.metadata["resolved_models"] = {k: sorted(v) for k, v in resolved.items()}
    _dump(result, out_dir)
    return result


def _dump(result: EvalResult, out_dir: Optional[Path]) -> None:
    if out_dir:
        (out_dir / "eval.json").write_text(json.dumps(result.to_json(), indent=2, default=str))
        try:
            (out_dir / "digest.md").write_text(digest(result, out_dir))
        except Exception as e:  # noqa: BLE001 — a reading aid, never a failed evaluation
            (out_dir / "digest.md").write_text(f"(digest failed: {type(e).__name__}: {e})\n")


def digest(result: EvalResult, out_dir: Path, max_rows: int = 400) -> str:
    """A per-episode index of the traces (what happened when), so a reader finds the right place in
    the full JSONL before reading it: every model call (clock, latency, output tokens, answer length,
    error) and every turn-game step (action, score change) or, for real-time games, the score changes.
    It summarises; the traces stay the evidence."""
    out = [f"# Episode digest — {result.game}", ""]
    if result.fatal:
        out.append(f"fatal: {result.fatal.splitlines()[-1][:300]}")
    for ep in result.episodes:
        out += [f"## seed {ep.seed}: score {ep.score:g}, {ep.steps} steps, ended by {ep.ended_by}, "
                f"{ep.wall_s:.0f} s wall, S2 calls {ep.system2_calls} (failed {ep.failed_s2}), S1 calls {ep.system1_calls}"
                + (f", stale ticks {ep.stale_ticks}/{ep.stale_ticks + ep.decided_ticks}" if ep.stale_ticks + ep.decided_ticks else "")]
        if ep.error:
            out.append(f"error: {_last_line(ep.error, 300)}")
        path = out_dir / f"trace_seed{ep.seed}.jsonl"
        if not path.exists():
            out.append("")
            continue
        rows, last_score, n = [], None, 0
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("type")
            if t == "model_call":
                resp = r.get("response") or {}
                text = resp.get("text") if isinstance(resp, dict) else None
                tok = (r.get("usage") or {}).get("output_tokens")
                rows.append(f"| {r.get('t_start', 0) - ep_start(path):.0f} | {r.get('kind')} call | "
                            f"{r.get('latency_s', 0):.1f} s" + (f", {tok} tok" if tok else "")
                            + (f", {len(text)} chars" if isinstance(text, str) else "")
                            + (f", ERROR {_last_line(r.get('error'), 80)}" if not r.get("ok") else "") + " | |")
            elif t == "step":
                score = r.get("score")
                delta = "" if last_score is None or score == last_score else f" ({score - last_score:+g})"
                last_score = score
                rows.append(f"| step {r.get('step')} | act `{str(r.get('action'))[:40]}` | score {score:g}{delta} | |")
            elif t == "step_error":
                rows.append(f"| | act `{str(r.get('action'))[:40]}` | ILLEGAL: {str(r.get('error'))[:100]} | |")
            elif t == "tick":
                score = r.get("score")
                if last_score is not None and score != last_score:
                    rows.append(f"| tick {r.get('tick')} | act `{str(r.get('action'))[:40]}` | score {score:g} ({score - last_score:+g}) | |")
                last_score = score
            elif t in ("deadline", "game_error", "harness") and n < 50:
                n += 1
                keep = {k: v for k, v in r.items() if k not in ("type", "t", "seed", "clock")}
                rows.append(f"| {t} | {json.dumps(keep, default=str)[:160]} | | |")
        if len(rows) > max_rows:
            rows = rows[: max_rows // 2] + [f"| … {len(rows) - max_rows} rows omitted … | | | |"] + rows[-max_rows // 2:]
        out += ["", "| when | what | result | |", "|---|---|---|---|"] + rows + [""]
    return "\n".join(out)


def ep_start(path: Path) -> float:
    """Wall time of the episode's first trace record (digest timestamps are relative to it)."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                return float(json.loads(line).get("t", 0.0))
            except ValueError:
                continue
    return 0.0


class _CountingSink(TraceSink):
    """Pre-flight trace: counts records instead of keeping them (a fast game can tick millions of times)."""

    def __init__(self) -> None:
        super().__init__(None)
        self.counts: collections.Counter = collections.Counter()

    def write(self, record) -> None:
        kind = record.get("type", "")
        if kind == "model_call":
            kind += ":" + str(record.get("kind")) + (":ok" if record.get("ok") else ":failed")
        self.counts[kind] += 1


def preflight_harness(harness_path: Path | str, game_factory: Any, *, game_kwargs: Optional[dict[str, Any]] = None,
                      tier2: bool = True, timeout_s: float = 60.0, seed: int = 0,
                      budget: Optional[Budget] = None, max_ticks: int = 2000) -> dict[str, Any]:
    """Cheap contract check before spending real model budget.

    One episode on the *virtual* clock with stand-in models (System-2 returns arbitrary text,
    System-1 the first option of every question), the evaluation's call budgets and a real-time
    guard of ``timeout_s``. Real-time games stop after ``max_ticks`` ticks, so a live game server
    is not flooded. Harness exceptions, unfinished background work and game failures fail the
    check; budget and deadline ends and a low score are fine. ``coverage`` tells how much of the
    harness actually ran (answered model calls, posted actions)."""
    from ..models.mock import HeuristicSystem1, ScriptedSystem2
    harness_path = Path(harness_path)
    try:
        module = load_harness_module(harness_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "load", "repairable": True, "error": _tb(e, 4000)}
    s2 = ScriptedSystem2(fn=lambda prompt: "I am not sure. Let us try something: maybe ACTION or {\"plan\": []}",
                         latency_s=1.0)
    s1 = HeuristicSystem1(latency_s=0.1) if tier2 else None
    try:
        game = game_factory(**(game_kwargs or {}))
    except Exception as e:  # noqa: BLE001 — an adaptor problem the proposer cannot repair
        unload_harness_module(module)
        return {"ok": False, "stage": "game", "repairable": False, "error": "game construction failed: " + _tb(e, 2000)}
    b = Budget(max_system2_calls=budget.max_system2_calls if budget else 6,
               max_system1_calls=budget.max_system1_calls if budget else 400, max_wall_s=timeout_s)
    sink = _CountingSink()
    tick_s = getattr(game, "tick_s", None)
    cap = max_ticks * float(tick_s) if hasattr(game, "tick") and tick_s else None
    try:
        ep = run_episode(module, game, seed=seed, system2=s2, system1=s1, trace=sink, memory={}, budget=b,
                         clock_mode="virtual", harness_timeout_s=timeout_s, grace_s=5.0, max_virtual_s=cap)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "episode", "repairable": True, "error": _tb(e, 4000)}
    finally:
        close_game(game)
        unload_harness_module(module)
    c = sink.counts
    coverage = {"system2_answers": c["model_call:system2:ok"], "system1_answers": c["model_call:system1:ok"],
                "failed_calls": c["model_call:system2:failed"] + c["model_call:system1:failed"],
                "acts": ep.acts, "acts_after_s1": ep.acts_after_s1, "acts_after_s2": ep.acts_after_s2,
                "steps": ep.steps, "harness_events": c["harness"]}
    warnings = []
    if tier2 and coverage["system1_answers"] == 0:
        warnings.append("no System-1 call was answered during the dry run (the stand-in answers every question)")
    if coverage["system2_answers"] == 0:
        warnings.append("no System-2 call was answered during the dry run")
    if ep.acts == 0:
        warnings.append("the harness never posted an action")
    err = ep.error or ""
    return {"ok": not err, "stage": "game" if ep.ended_by in ("game_error", "reset_failed") else "episode",
            # a reset failure is the adaptor's; a tick failure names the action the harness posted
            "repairable": ep.ended_by != "reset_failed", "error": err, "ended_by": ep.ended_by,
            "score": ep.score, "steps": ep.steps, "system1_calls": ep.system1_calls,
            "system2_calls": ep.system2_calls, "coverage": coverage, "warnings": warnings}
