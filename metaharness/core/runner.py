"""Evaluate one harness candidate on one game over several seeds.

Produces per-episode scores plus the *latency accounting* that makes tier-2
harness comparisons honest: how many ticks ran on a stale action, model
latency distribution, and how many System-1/2 calls were spent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..models.base import ModelUnavailable, System1Model, System2Model, TraceSink
from .contract import Budget, BudgetExceeded, HarnessContext, VirtualClock, WallClock, load_harness_module


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

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_s1"] = _stats(self.latency_s1)
        d["latency_s2"] = _stats(self.latency_s2)
        return d


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": xs[len(xs) // 2],
            "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}


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
            "errors": [e.error for e in self.episodes if e.error],
            "episodes": [e.summary() for e in self.episodes],
            "metadata": self.metadata,
            "score_std": (statistics.stdev(e.score for e in self.episodes)
                          if len(self.episodes) > 1 else None),
        }


class _GameView:
    """What the harness sees: observations + metadata, never the stepping API."""

    def __init__(self, game: Any, stopped: threading.Event) -> None:
        self._g = game
        self._stopped = stopped
        self.name = game.name
        self.action_spec = game.action_spec
        self.tick_s = getattr(game, "tick_s", None)

    def observe(self) -> Any:
        return self._g.observe()

    def describe(self) -> str:
        return self._g.describe()

    @property
    def done(self) -> bool:
        return self._stopped.is_set() or self._g.done

    @property
    def score(self) -> float:
        return self._g.score

    @property
    def ticks(self) -> int:
        return getattr(self._g, "ticks", 0)

    def info(self) -> dict[str, Any]:
        # Game.info() is evaluator diagnostics and can contain hidden rules.
        return {"score": self.score, "ticks": self.ticks, "game_done": self._g.done}


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
) -> EpisodeResult:
    realtime = hasattr(game, "tick")
    game.reset(seed)
    stop = threading.Event()
    view = _GameView(game, stop)
    accepting_trace = threading.Event()
    accepting_trace.set()
    t_wall0 = time.time()
    res = EpisodeResult(seed=seed, score=0.0, steps=0, system1_calls=0, system2_calls=0)
    s1_lat: list[float] = []
    s2_lat: list[float] = []

    # tracing hooks for latency (trace records carry latency_s)
    class _Tee(TraceSink):
        def __init__(self, inner: TraceSink) -> None:
            self.inner = inner
            self.records = inner.records
            self.path = inner.path
            self._lock = threading.Lock()
            self._fh = None

        def write(self, record):
            if not accepting_trace.is_set():
                return
            r = dict(record)
            r["seed"] = seed
            if r.get("type") == "model_call" and r.get("ok"):
                (s1_lat if r.get("kind") == "system1" else s2_lat).append(float(r.get("latency_s", 0.0)))
            self.inner.write(r)

    tee = _Tee(trace)
    stale = decided = 0
    ticker: Optional[threading.Thread] = None
    mailbox = _Mailbox(game.action_spec.noop if realtime else None)

    if realtime:
        def _apply_tick() -> None:
            nonlocal stale, decided
            a, fresh = mailbox.take()
            if fresh:
                decided += 1
            else:
                stale += 1
            game.tick(a)
            tee.write({"type": "tick", "tick": game.ticks, "action": a, "fresh": fresh, "score": game.score})

        if clock_mode == "virtual":
            clock = VirtualClock()
            def _advance(t0: float, t1: float) -> None:
                target = int(t1 / game.tick_s + 1e-9)
                for _ in range(max(0, target - game.ticks)):
                    if stop.is_set() or game.done:
                        break
                    _apply_tick()
            clock.on_advance = _advance
        else:
            clock = WallClock()
            def _ticker() -> None:
                # phase-aligned: tick n fires at origin + n * tick_s
                n = 1
                while not stop.is_set() and not game.done:
                    delay = clock.origin + n * game.tick_s - time.time()
                    if delay > 0:
                        stop.wait(delay)
                    if stop.is_set() or game.done:
                        break
                    _apply_tick()
                    n += 1
            ticker = threading.Thread(target=_ticker, name="game-ticker", daemon=True)

        act_fn = mailbox.post
    else:
        clock = VirtualClock() if clock_mode == "virtual" else WallClock()
        def act_fn(action: Any) -> Any:
            if stop.is_set():
                return game.observe()
            obs = game.step(action)
            tee.write({"type": "step", "action": action, "score": game.score})
            return obs

    ctx = HarnessContext(system2=system2, system1=system1, trace=tee, clock=clock, budget=budget,
                         memory=memory, game_kind="realtime" if realtime else "turn",
                         act_fn=act_fn, observe_fn=game.observe, done_fn=lambda: game.done,
                         latency_override=latency_override,
                         tick_s=getattr(game, "tick_s", None) if realtime else None,
                         ticks_fn=(lambda: game.ticks) if realtime else None)
    tee.write({"type": "episode_start", "game": game.name, "realtime": realtime, "clock": clock_mode})
    err: Optional[str] = None
    try:
        if ticker:
            clock.origin = time.time()
            ticker.start()
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
        th.start()
        deadline_s = max(0.0, min(harness_timeout_s, budget.remaining()["wall_s"]))
        if not done_flag.wait(deadline_s):
            err = f"harness did not return within {deadline_s:.3f}s (episode deadline)"
        elif exc:
            e = exc[0]
            if isinstance(e, ModelUnavailable):
                raise e
            if isinstance(e, BudgetExceeded):
                err = f"budget exceeded: {e}"
            else:
                err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
    finally:
        stop.set()
        if ticker and ticker.is_alive():
            ticker.join(timeout=2.0)
        pending = ctx.close()
        accepting_trace.clear()
    if ctx.unavailable is not None:
        raise ctx.unavailable
    if pending and err is None:
        err = f"harness returned with {pending} unfinished background task(s)"
    res.score = float(game.score)
    res.steps = game.ticks if realtime else int(game.info().get("actions_used", 0))
    res.system1_calls = budget.system1_calls
    res.system2_calls = budget.system2_calls
    res.stale_ticks, res.decided_ticks = stale, decided
    res.latency_s1, res.latency_s2 = s1_lat, s2_lat
    res.wall_s = time.time() - t_wall0
    res.error = err
    res.info = {**view.info(), "pending_tasks": pending}
    trace.write({"type": "episode_end", **res.summary()})
    return res


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
) -> EvalResult:
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
    result = EvalResult(harness_path=str(harness_path), game="unknown", episodes=[])
    b = budget_per_episode or Budget()
    result.metadata = {
        "protocol_version": 2, "seeds": seeds, "game_kwargs": game_kwargs or {},
        "clock_mode": clock_mode, "latency_override": latency_override,
        "memory_mode": memory_mode, "harness_timeout_s": harness_timeout_s,
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
        module = load_harness_module(harness_path)
    except Exception as e:  # noqa: BLE001
        result.fatal = "load error: " + "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
        _dump(result, out_dir)
        return result
    memory: dict[str, Any] = {}
    for i, seed in enumerate(seeds):
        # Fresh game/module prevent timed-out workers and globals contaminating another seed.
        game = game_factory(**(game_kwargs or {}))
        result.game = game.name
        result.metadata["game_kind"] = "realtime" if hasattr(game, "tick") else "turn"
        if i:
            module = load_harness_module(harness_path)
        memory = copy.deepcopy(memory) if memory_mode == "shared" else {}
        trace = TraceSink(out_dir / f"trace_seed{seed}.jsonl" if out_dir else None, append=False)
        budget = Budget(max_system2_calls=b.max_system2_calls, max_system1_calls=b.max_system1_calls,
                        max_wall_s=b.max_wall_s)
        try:
            ep = run_episode(module, game, seed=seed, system2=system2, system1=system1, trace=trace,
                             memory=memory, budget=budget, clock_mode=clock_mode,
                             latency_override=latency_override, harness_timeout_s=harness_timeout_s)
            result.episodes.append(ep)
        except ModelUnavailable as e:
            result.fatal = f"model unavailable: {e}"
            trace.write({"type": "episode_unavailable", "seed": seed, "error": str(e)})
            break
        finally:
            trace.close()
    _dump(result, out_dir)
    return result


def _dump(result: EvalResult, out_dir: Optional[Path]) -> None:
    if out_dir:
        (out_dir / "eval.json").write_text(json.dumps(result.to_json(), indent=2, default=str))
