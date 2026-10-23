"""Shared subprocess plumbing for CLI-backed System-2 models."""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import os
import signal
import subprocess
import threading
import time
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


@contextlib.contextmanager
def cancel_scope(check: Callable[[], Optional[str]]) -> Iterator[None]:
    """CLI calls made by this thread inside the block are killed once ``check()`` returns a reason."""
    token = _CANCEL.set(check)
    try:
        yield
    finally:
        _CANCEL.reset(token)


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
    """Kill every CLI invocation still running (interpreter exit)."""
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
        if time.monotonic() >= deadline:
            _signal_kill(proc)
            th.join(10)
            out, err = box.get("out") or ("", "")
            raise CLITimeout(f"CLI invocation exceeded {timeout_s:.3f}s", out or "", err or "")
        reason = check()
        if reason:
            _signal_kill(proc)
            th.join(10)
            raise ModelCallCancelled(f"model call cancelled: {reason}")
    if "exc" in box:
        raise box["exc"]  # type: ignore[misc]
    out, err = box["out"]  # type: ignore[misc]
    return out, err
