"""Shared subprocess plumbing for CLI-backed System-2 models."""
from __future__ import annotations

import contextlib
import contextvars
import os
import signal
import subprocess
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


def _kill(proc: subprocess.Popen) -> tuple[str, str]:
    """Kill the whole invocation (tool subprocesses included) and collect its remaining output."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        out, err = proc.communicate(timeout=10)
    except Exception:  # noqa: BLE001 — a detached grandchild may hold the pipes open
        return "", ""
    return out or "", err or ""


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
    check = _CANCEL.get()
    deadline = time.monotonic() + timeout_s
    pending_input = stdin_text
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out, err = _kill(proc)
                raise CLITimeout(f"CLI invocation exceeded {timeout_s:.3f}s", out, err)
            try:
                # communicate() may be retried after TimeoutExpired without losing output;
                # the input is only passed on the first call.
                stdout, stderr = proc.communicate(input=pending_input,
                                                  timeout=min(remaining, POLL_S) if check else remaining)
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                reason = check() if check else None
                if reason:
                    _kill(proc)
                    raise ModelCallCancelled(f"model call cancelled: {reason}") from None
    except BaseException:
        # KeyboardInterrupt / SystemExit / cancellation: never leave the agent running behind us.
        if proc.poll() is None:
            _kill(proc)
        raise
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)
