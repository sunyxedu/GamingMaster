"""Any Claude model through the ``claude`` CLI (``claude -p --output-format json``).

Which model is a config concern (``metaharness/models.toml``); this class only
drives the CLI. Same interface and session semantics (``--resume``) as the
codex backend. Never substituted silently for another profile.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ._cli import run_cli
from .base import ModelCall, ModelUnavailable, System2Result, TraceSink


class ClaudeCLI:
    def __init__(self, model: Optional[str] = None, reasoning_effort: Optional[str] = None,
                 claude_bin: str = "claude") -> None:
        self.model = model
        self.name = f"claude:{model or 'default'}"
        self.reasoning_effort = reasoning_effort      # accepted for profile symmetry; claude -p has no flag
        self.claude_bin = claude_bin
        self.n_calls = 0
        self.total_cost_usd = 0.0

    def run(
        self,
        prompt: str,
        *,
        cwd: Optional[Path] = None,
        session_id: Optional[str] = None,
        system: Optional[str] = None,
        agentic: bool = False,
        timeout_s: float = 600.0,
        trace: Optional[TraceSink] = None,
        images: Optional[list[str]] = None,
        full_access: bool = False,
        web_search: bool = False,
    ) -> System2Result:
        argv = [self.claude_bin, "-p", "--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]
        if session_id:
            argv += ["--resume", session_id]
        if system and not session_id:
            argv += ["--system-prompt", system]
        if agentic:
            argv += ["--permission-mode", "bypassPermissions", "--dangerously-skip-permissions"]
        else:
            argv += ["--tools", ""]  # plain answer, no tool use
        cwd = Path(cwd).resolve() if cwd else None
        t0 = time.time()
        # Nested Claude Code sessions refuse to start unless this is cleared.
        proc = run_cli(argv, cwd=cwd, stdin_text=prompt, timeout_s=timeout_s,
                       env_drop=("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"))
        latency = time.time() - t0
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        if proc.returncode != 0 or payload.get("is_error"):
            err = (payload.get("result") or proc.stderr or proc.stdout)[-1500:]
            if trace:
                trace.log_call(ModelCall("system2", "claude", t0, latency, {"prompt": prompt}, None, ok=False, error=err))
            if "rate limit" in err.lower() or "usage limit" in err.lower() or "not logged in" in err.lower():
                raise ModelUnavailable(err)
            raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {err}")
        text = payload.get("result", "") or ""
        usage = {"total_cost_usd": payload.get("total_cost_usd"), **(payload.get("usage") or {})}
        self.n_calls += 1
        self.total_cost_usd += float(payload.get("total_cost_usd") or 0.0)
        res = System2Result(text=text, session_id=payload.get("session_id") or session_id, latency_s=latency, usage=usage, raw=payload)
        if trace:
            trace.log_call(ModelCall("system2", self.name, t0, latency,
                                     {"prompt": prompt}, {"text": text, "session_id": res.session_id}, usage=usage))
        return res
