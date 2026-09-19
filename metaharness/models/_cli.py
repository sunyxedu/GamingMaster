"""Shared subprocess plumbing for CLI-backed System-2 models."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Optional, Sequence


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
    # Kill the entire invocation on timeout, including tool subprocesses.
    proc = subprocess.Popen(
        list(argv), cwd=str(cwd) if cwd else None,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.communicate()
        raise TimeoutError(f"CLI invocation exceeded {timeout_s:.3f}s") from None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)
