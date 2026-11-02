"""Shared subprocess plumbing for CLI-backed System-2 models."""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import os
import random
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from .base import ModelCallCancelled

# A game-player call (HarnessContext.think) runs inside a cancel scope: ``check()`` returns a
# reason once the call's episode is over, and run_cli then kills the CLI process group instead
# of letting it outlive the episode (quota, CPU and latency of the next seed).
_CANCEL: contextvars.ContextVar[Optional[Callable[[], Optional[str]]]] = contextvars.ContextVar(
    "metaharness_cli_cancel", default=None)
POLL_S = 0.2

# Every CLI child runs in its own session (so a timeout kills its tool processes too), which also
# means it survives our own exit: track them and kill what is left when the interpreter exits
# (SIGTERM / Ctrl-C unwind through SystemExit / KeyboardInterrupt, so atexit runs).
_LIVE: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()
# Set once the process is shutting down (SIGTERM / Ctrl-C): a CLI call that fails from now on was
# killed by us, and callers must not record it as the model's failure.
STOPPING = threading.Event()


@contextlib.contextmanager
def cancel_scope(check: Callable[[], Optional[str]]) -> Iterator[None]:
    """CLI calls made by this thread inside the block are killed once ``check()`` returns a reason."""
    token = _CANCEL.set(check)
    try:
        yield
    finally:
        _CANCEL.reset(token)


@contextlib.contextmanager
def slot(directory: Optional[Path], pool: str, n: int, since: Optional[int] = None) -> Iterator[None]:
    """Hold one of ``n`` slots of ``pool`` (lock files in ``directory`` shared by every process of this
    user) for the block; ``n <= 0``: no cap. A shared per-user request limit makes the CLI processes of
    every search on the machine compete, so agents and players get separate pools, and waiters are
    served in arrival order (ticket files in ``<pool>.queue/``; ``since``, a call's first arrival in ns,
    keeps a retry's place): without the queue a waiter could lose every race for hours. Waiting
    honours the calling episode's cancel scope."""
    try:
        import fcntl
    except ImportError:          # not POSIX: no cross-process cap
        n = 0
    if n <= 0 or directory is None:
        yield
        return
    queue = directory / f"{pool}.queue"
    queue.mkdir(parents=True, exist_ok=True)
    ticket = queue / f"{since or time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    ticket.touch()
    handles = [open(directory / f"{pool}{i}.lock", "a") for i in range(n)]
    held = None
    try:
        while held is None:
            if _first_in_line(queue, ticket.name):
                for fh in handles:
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held = fh
                        break
                    except OSError:
                        continue
            if held is None:
                check = _CANCEL.get()
                reason = check() if check else None
                if reason:
                    raise ModelCallCancelled(f"model call cancelled while waiting for a slot: {reason}")
                time.sleep(0.1 + random.random() * 0.2)
        ticket.unlink(missing_ok=True)
        yield
    finally:
        ticket.unlink(missing_ok=True)
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
        for fh in handles:
            fh.close()


def _first_in_line(queue: Path, name: str) -> bool:
    """No live ticket older than ``name`` (tickets of processes that died are removed)."""
    for t in sorted(os.listdir(queue)):
        if t >= name:
            return True
        try:
            os.kill(int(t.split("-")[1]), 0)
            return False
        except ProcessLookupError:
            with contextlib.suppress(OSError):
                (queue / t).unlink()
        except (ValueError, IndexError):
            with contextlib.suppress(OSError):
                (queue / t).unlink()
        except PermissionError:   # alive (another user's process)
            return False
    return True


class CLITimeout(TimeoutError):
    """The CLI exceeded its deadline. Carries what it printed before it was killed, so a backend
    can still recover e.g. the session id of a coding agent that ran out of time."""

    def __init__(self, message: str, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _signal_kill(proc: subprocess.Popen) -> None:
    """Kill the whole invocation, tool subprocesses included."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill(proc: subprocess.Popen) -> tuple[str, str]:
    """Kill and collect the remaining output (only when no other thread is in communicate())."""
    _signal_kill(proc)
    try:
        out, err = proc.communicate(timeout=10)
    except Exception:  # noqa: BLE001 — a detached grandchild may hold the pipes open
        return "", ""
    return out or "", err or ""


def kill_all() -> None:
    """Kill every CLI invocation still running (interpreter exit, stop of the process)."""
    STOPPING.set()
    with _LIVE_LOCK:
        procs = list(_LIVE)
    for proc in procs:
        if proc.poll() is None:
            _signal_kill(proc)


atexit.register(kill_all)


def run_cli(
    argv: Sequence[str],
    *,
    cwd: Optional[Path],
    stdin_text: Optional[str],
    timeout_s: float,
    env_overrides: Optional[dict[str, str]] = None,
    env_drop: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for k in env_drop:
        env.pop(k, None)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        list(argv), cwd=str(cwd) if cwd else None,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        start_new_session=(os.name == "posix"),
    )
    with _LIVE_LOCK:
        _LIVE.add(proc)
    check = _CANCEL.get()
    try:
        if check is None:
            try:
                stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                out, err = _kill(proc)
                raise CLITimeout(f"CLI invocation exceeded {timeout_s:.3f}s", out, err) from None
        else:
            stdout, stderr = _communicate_polled(proc, stdin_text, timeout_s, check)
    except BaseException:
        # KeyboardInterrupt / SystemExit / cancellation: never leave the agent running behind us.
        if proc.poll() is None:
            _signal_kill(proc)
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.discard(proc)
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def _communicate_polled(proc: subprocess.Popen, stdin_text: Optional[str], timeout_s: float,
                        check: Callable[[], Optional[str]]) -> tuple[str, str]:
    """communicate() on a helper thread, polling the cancel check here. It must not be split into
    short communicate() calls: once one times out with input pending, CPython never writes the
    rest of stdin, so a prompt larger than the pipe buffer would never reach the CLI."""
    box: dict[str, object] = {}

    def talk() -> None:
        try:
            box["out"] = proc.communicate(input=stdin_text)
        except BaseException as e:  # noqa: BLE001 — re-raised in the calling thread
            box["exc"] = e

    th = threading.Thread(target=talk, name="cli-communicate", daemon=True)
    th.start()
    deadline = time.monotonic() + timeout_s
    while True:
        th.join(max(0.0, min(POLL_S, deadline - time.monotonic())))
        if not th.is_alive():
            break
        reason = check()       # first: a call cancelled at the episode deadline is not a timeout
        if reason:
            _signal_kill(proc)
            th.join(10)
            raise ModelCallCancelled(f"model call cancelled: {reason}")
        if time.monotonic() >= deadline:
            _signal_kill(proc)
            th.join(10)
            out, err = box.get("out") or ("", "")
            raise CLITimeout(f"CLI invocation exceeded {timeout_s:.3f}s", out or "", err or "")
    if "exc" in box:
        raise box["exc"]  # type: ignore[misc]
    out, err = box["out"]  # type: ignore[misc]
    return out, err
