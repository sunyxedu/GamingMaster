"""Any OpenAI model through the ``codex`` CLI, non-interactively.

Which model (Astra, Sol, ...) is a *config* concern: see ``metaharness/models.toml``
(``[system2.astra] backend = "codex", model = "gpt-6-astra"``). This class only
knows how to drive the CLI.

Mirrors Auto-CTFer's codex backend (engine/src/solvers/codex.ts): the CLI's own
login (``~/.codex/auth.json``) is used, ``OPENAI_API_KEY``/``OPENAI_BASE_URL``
are dropped from the child env so the CLI never silently switches auth, and the
reasoning effort is passed as ``model_reasoning_effort``.

``codex exec --json`` prints JSONL events; the final agent message is also
written with ``-o``. ``codex exec resume <thread-id> <prompt>`` continues a
thread so Astra's hidden reasoning state persists across game turns — the
setting behind OpenAI's ARC-AGI-3 result.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from ._cli import run_cli
from .base import ModelCall, ModelUnavailable, System2Result, TraceSink

_USAGE_LIMIT_RE = re.compile(r"usage limit|insufficient_quota|purchase more credits", re.I)
_AUTH_RE = re.compile(r"not logged in|login required|401|unauthorized", re.I)


class CodexCLI:
    def __init__(
        self,
        model: str,
        reasoning_effort: Optional[str] = None,
        codex_bin: str = "codex",
        ephemeral: bool = False,
        extra_config: Optional[dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.name = f"codex:{model}"
        self.reasoning_effort = reasoning_effort
        self.codex_bin = codex_bin
        self.ephemeral = ephemeral
        self.extra_config = dict(extra_config or {})
        self.n_calls = 0
        self.total_usage: dict[str, int] = {}

    # -- argv -----------------------------------------------------------
    def _argv(self, prompt: str, *, cwd: Optional[Path], session_id: Optional[str],
              agentic: bool, last_msg_path: Path) -> list[str]:
        argv = [self.codex_bin, "exec"]
        if session_id:
            argv += ["resume", session_id]
        # `resume` accepts a subset of `exec` flags (no --color / -s / -C).
        argv += ["--json", "--skip-git-repo-check", "-o", str(last_msg_path)]
        if agentic:
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        if not session_id:
            argv += ["-m", self.model, "--color", "never"]
            if not agentic:
                argv += ["-s", "read-only"]
            if cwd:
                argv += ["-C", str(cwd)]
            if self.ephemeral:
                argv += ["--ephemeral"]
        cfg = dict(self.extra_config)
        if self.reasoning_effort:
            cfg["model_reasoning_effort"] = f'"{self.reasoning_effort}"'
        # Silence hooks/notifiers from the user's interactive config.
        cfg.setdefault("notify", "[]")
        for k, v in cfg.items():
            argv += ["-c", f"{k}={v}"]
        argv.append(prompt)
        return argv

    # -- run ------------------------------------------------------------
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
    ) -> System2Result:
        full_prompt = f"{system}\n\n{prompt}" if system and not session_id else prompt
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            last_msg = Path(td) / "last.txt"
            argv = self._argv(full_prompt, cwd=cwd, session_id=session_id,
                              agentic=agentic, last_msg_path=last_msg)
            try:
                proc = run_cli(argv, cwd=cwd, stdin_text=None, timeout_s=timeout_s,
                               env_drop=("OPENAI_API_KEY", "OPENAI_BASE_URL"))
            except Exception as e:  # timeout etc.
                self._trace(trace, t0, full_prompt, None, str(e))
                raise
            text = last_msg.read_text() if last_msg.exists() else ""
        events = _parse_jsonl(proc.stdout)
        thread_id, usage, errors = _digest_events(events)
        stderr_tail = (proc.stderr or "")[-2000:]
        combined_err = "\n".join(errors + ([stderr_tail] if proc.returncode != 0 else []))
        if _USAGE_LIMIT_RE.search(combined_err) or _USAGE_LIMIT_RE.search(proc.stdout[-4000:]):
            self._trace(trace, t0, full_prompt, None, combined_err)
            raise ModelUnavailable(f"codex usage limit: {combined_err.strip()[:300]}")
        if proc.returncode != 0 and not text:
            self._trace(trace, t0, full_prompt, None, combined_err)
            if _AUTH_RE.search(combined_err):
                raise ModelUnavailable(f"codex auth problem: {combined_err[:300]}")
            raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {combined_err[:800]}")
        if not text:  # fall back to the last agent_message event
            text = _last_agent_message(events)
        latency = time.time() - t0
        self.n_calls += 1
        for k, v in usage.items():
            if isinstance(v, int):
                self.total_usage[k] = self.total_usage.get(k, 0) + v
        res = System2Result(text=text, session_id=thread_id or session_id, latency_s=latency, usage=usage, raw=events)
        self._trace(trace, t0, full_prompt, res, None)
        return res

    def _trace(self, trace: Optional[TraceSink], t0: float, prompt: str, res: Optional[System2Result], err: Optional[str]) -> None:
        if not trace:
            return
        trace.log_call(ModelCall(
            "system2", self.name, t0, time.time() - t0,
            {"prompt": prompt}, {"text": res.text, "session_id": res.session_id} if res else None,
            ok=err is None, error=err, usage=res.usage if res else {},
        ))


def _parse_jsonl(stdout: str) -> list[dict]:
    out: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _digest_events(events: list[dict]) -> tuple[Optional[str], dict, list[str]]:
    thread_id: Optional[str] = None
    usage: dict = {}
    errors: list[str] = []
    for ev in events:
        t = ev.get("type", "")
        if t == "thread.started":
            thread_id = ev.get("thread_id") or ev.get("id") or thread_id
        elif t == "turn.completed":
            usage = ev.get("usage") or usage
        elif t in ("error", "turn.failed"):
            msg = ev.get("message") or (ev.get("error") or {}).get("message") or json.dumps(ev)
            errors.append(str(msg))
    return thread_id, usage, errors


def _last_agent_message(events: list[dict]) -> str:
    text = ""
    for ev in events:
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text", "") or text
    return text
