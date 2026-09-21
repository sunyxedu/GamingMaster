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
thread so Astra's conversation context persists across game turns. This does not establish
persistence of inaccessible model-internal reasoning state.
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
        self.player_tools = "disabled-and-audited"

    # -- argv -----------------------------------------------------------
    def _argv(self, prompt: str, *, cwd: Optional[Path], session_id: Optional[str],
              agentic: bool, last_msg_path: Path, images: Optional[list[str]] = None,
              full_access: bool = False, web_search: bool = False) -> list[str]:
        argv = [self.codex_bin, "exec"]
        if session_id:
            argv += ["resume", session_id]
        for img in images or []:              # screenshots for visual games
            argv += ["-i", str(img)]
        # `resume` accepts a subset of `exec` flags (no --color / -s / -C).
        argv += ["--json", "--skip-git-repo-check", "-o", str(last_msg_path)]
        argv += ["-m", self.model]
        sandbox = "danger-full-access" if (agentic and full_access) else ("workspace-write" if agentic else "read-only")
        if not session_id:
            argv += ["--color", "never", "-s", sandbox]
            if cwd:
                argv += ["-C", str(cwd)]
            if self.ephemeral:
                argv += ["--ephemeral"]
        cfg = dict(self.extra_config)
        # resume inherits a session's permissions unless explicitly overridden.
        cfg["sandbox_mode"] = f'"{sandbox}"'
        cfg["approval_policy"] = '"never"'
        if agentic:
            # Study / adapt phases research the game on the web; the proposer must not.
            cfg["web_search"] = '"live"' if web_search else '"disabled"'
        # Never inherit the user's interactive speed tier: experiments run on the
        # standard tier so latency numbers are comparable and credits are not 2x.
        cfg.setdefault("service_tier", '"default"')
        cfg.setdefault("features.fast_mode", "false")
        if not agentic:
            # The player must not pick up the user's global AGENTS.md / project docs.
            cfg["project_doc_max_bytes"] = "0"
            # A game player is inference-only. Filesystem/search access would
            # expose hidden simulator state or answers outside its observation.
            cfg["features.shell_tool"] = "false"
            cfg["features.apps"] = "false"
            cfg["features.browser_use"] = "false"
            cfg["features.computer_use"] = "false"
            cfg["features.multi_agent"] = "false"
            cfg["web_search"] = '"disabled"'
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
        images: Optional[list[str]] = None,
        full_access: bool = False,
        web_search: bool = False,
    ) -> System2Result:
        """``full_access`` (agentic only) lifts the workspace sandbox so the model can
        install packages / start game servers; ``web_search`` enables live search."""
        full_prompt = f"{system}\n\n{prompt}" if system and not session_id else prompt
        if not agentic:
            full_prompt = ("Inference-only game player: use only the supplied observations and "
                           "conversation. Do not use tools, read files, run commands, browse, or "
                           "consult external memories. Return the requested answer directly.\n\n"
                           + full_prompt)
        cwd = Path(cwd).resolve() if cwd else None      # -C and subprocess cwd must both be absolute
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            last_msg = Path(td) / "last.txt"
            argv = self._argv(full_prompt, cwd=cwd, session_id=session_id,
                              agentic=agentic, last_msg_path=last_msg, images=images,
                              full_access=full_access, web_search=web_search)
            try:
                proc = run_cli(argv, cwd=cwd, stdin_text=None, timeout_s=timeout_s,
                               env_drop=("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
                                         "TYPESAFE_API_KEY"))
            except Exception as e:  # timeout etc.
                self._trace(trace, t0, full_prompt, None, str(e))
                raise
            text = last_msg.read_text() if last_msg.exists() else ""
        events = _parse_jsonl(proc.stdout)
        thread_id, usage, errors = _digest_events(events)
        if not agentic:
            external = [ev.get("item", {}).get("type") for ev in events
                        if ev.get("type") in ("item.started", "item.completed")
                        and ev.get("item", {}).get("type") in
                        {"command_execution", "mcp_tool_call", "web_search", "file_change",
                         "collab_tool_call"}]
            if external:
                error = "inference-only player used external tools: " + ", ".join(sorted(set(external)))
                self._trace(trace, t0, full_prompt, None, error)
                raise ModelUnavailable(error)
        stderr_tail = (proc.stderr or "")[-2000:]
        combined_err = "\n".join(errors + ([stderr_tail] if proc.returncode != 0 else []))
        if _USAGE_LIMIT_RE.search(combined_err) or _USAGE_LIMIT_RE.search(proc.stdout[-4000:]):
            self._trace(trace, t0, full_prompt, None, combined_err)
            raise ModelUnavailable(f"codex usage limit: {combined_err.strip()[:300]}")
        if proc.returncode != 0 or errors:
            self._trace(trace, t0, full_prompt, None, combined_err)
            if _AUTH_RE.search(combined_err):
                raise ModelUnavailable(f"codex auth problem: {combined_err[:300]}")
            raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {combined_err[:800]}")
        if not text:  # fall back to the last agent_message event
            text = _last_agent_message(events)
        if not text.strip():
            self._trace(trace, t0, full_prompt, None, "empty completed response")
            raise RuntimeError("codex returned no final response")
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
