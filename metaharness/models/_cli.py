"""Shared subprocess plumbing for CLI-backed System-2 models."""
from __future__ import annotations

import os
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
    return subprocess.run(
        list(argv), cwd=str(cwd) if cwd else None, input=stdin_text,
        capture_output=True, text=True, timeout=timeout_s, env=env, check=False,
    )
