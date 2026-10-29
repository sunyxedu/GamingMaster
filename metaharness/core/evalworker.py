"""Evaluate one candidate on one game in a child process.

    python -m metaharness.core.evalworker job.json

The search evaluates several candidates at once. A process each keeps their harness threads, game
clocks and CLI children apart (a busy harness cannot delay another evaluation's ticks through the
GIL, nothing leaks between modules) and lets a crashed evaluation fail alone. The job names the
models by profile and the game by registry key or adaptor directory, so the child builds exactly
what the parent would have used; the result is the eval.json ``evaluate_harness`` writes.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import REPO_ROOT


@dataclass
class EvalJob:
    harness_path: str
    out_dir: str
    game: str                                   # registry key (test bed) or the adaptor's name
    seeds: list[int]
    adaptor_dir: Optional[str] = None           # register_adaptor(adaptor_dir) first
    game_kwargs: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)   # system1 / system2 profile names, effort, tier
    clock_mode: str = "wall"
    budget: dict[str, Any] = field(default_factory=dict)   # Budget fields
    harness_timeout_s: float = 600.0
    memory_mode: str = "episode"
    grace_s: float = 10.0
    latency_override: Optional[dict[str, float]] = None
    game_name: Optional[str] = None

    @property
    def eval_path(self) -> Path:
        return Path(self.out_dir) / "eval.json"


def run_job(job: EvalJob) -> dict[str, Any]:
    """Evaluate in this process; returns the eval dict (also written to ``out_dir/eval.json``)."""
    from ..cli import _system1_for, _system2
    from ..config import load_dotenv
    from ..games.registry import make_game, register_adaptor
    from .contract import Budget
    from .runner import evaluate_harness
    load_dotenv()
    key = register_adaptor(Path(job.adaptor_dir), kwargs=job.game_kwargs or None) if job.adaptor_dir else job.game
    m = job.models
    res = evaluate_harness(
        job.harness_path, lambda **k: make_game(key, **k), list(job.seeds),
        system2=_system2(m.get("system2") or "none", m.get("effort")),
        system1=_system1_for(m.get("system1") or "none", int(m.get("tier") or 2)),
        out_dir=Path(job.out_dir), clock_mode=job.clock_mode,
        budget_per_episode=Budget(**job.budget) if job.budget else None,
        latency_override=job.latency_override, harness_timeout_s=job.harness_timeout_s,
        game_kwargs=job.game_kwargs, memory_mode=job.memory_mode, grace_s=job.grace_s,
        game_name=job.game_name or job.game)
    return res.to_json()


def _fatal(job: EvalJob, text: str) -> dict[str, Any]:
    e = {"harness_path": job.harness_path, "game": job.game_name or job.game, "fatal": text, "mean_score": None,
         "n_episodes": 0, "errors": [], "episodes": [], "metadata": {"seeds": list(job.seeds)}}
    Path(job.out_dir).mkdir(parents=True, exist_ok=True)
    job.eval_path.write_text(json.dumps(e, indent=2))
    return e


def main(argv: Optional[list[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    job = EvalJob(**json.loads(Path(args[0]).read_text()))
    try:
        run_job(job)
    except BaseException as e:  # noqa: BLE001 — the parent reads why from eval.json
        if not job.eval_path.exists():
            _fatal(job, "evaluator crashed: " + "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:])
        return 1
    return 0


# -- parent side ------------------------------------------------------------------------------
_CHILDREN: set[subprocess.Popen] = set()
_LOCK = threading.Lock()


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def kill_children() -> None:
    """Kill every evaluation child still running (interpreter exit, Ctrl-C)."""
    with _LOCK:
        procs = list(_CHILDREN)
    for proc in procs:
        if proc.poll() is None:
            _kill(proc)


atexit.register(kill_children)


def _run_child(job: EvalJob) -> dict[str, Any]:
    out = Path(job.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if job.eval_path.exists():                  # never mistake a stale result for this run's
        job.eval_path.unlink()
    spec = out / "evalworker.json"
    spec.write_text(json.dumps(asdict(job), indent=2))
    log_path = out / "evalworker.log"
    with open(log_path, "w") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "metaharness.core.evalworker", str(spec)], cwd=str(REPO_ROOT),
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=(os.name == "posix"))
        with _LOCK:
            _CHILDREN.add(proc)
        try:
            proc.wait()
        except BaseException:
            _kill(proc)
            raise
        finally:
            with _LOCK:
                _CHILDREN.discard(proc)
    if job.eval_path.exists():
        try:
            return json.loads(job.eval_path.read_text())
        except ValueError as e:
            return _fatal(job, f"evaluator wrote an unreadable eval.json ({e})")
    tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else ""
    return _fatal(job, f"evaluator exited with status {proc.returncode} without a result\n{tail}")


def run_jobs(jobs: list[EvalJob], max_workers: int, on_done: Optional[Callable[[EvalJob, dict[str, Any]], None]] = None
             ) -> list[dict[str, Any]]:
    """Run the jobs in child processes, at most ``max_workers`` at a time; results in job order.
    ``on_done`` is called from the worker thread as each job finishes. An interrupt kills the children."""
    if not jobs:
        return []
    ex = cf.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs))))

    def one(job: EvalJob) -> dict[str, Any]:
        e = _run_child(job)
        if on_done is not None:
            on_done(job, e)
        return e

    futures = [ex.submit(one, j) for j in jobs]
    try:
        return [f.result() for f in futures]
    except BaseException:
        kill_children()
        raise
    finally:
        ex.shutdown(wait=True)


if __name__ == "__main__":
    sys.exit(main())
