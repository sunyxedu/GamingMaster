"""Any OpenAI model through the ``codex`` CLI, non-interactively.

Which model (Astra, Sol, ...) is a *config* concern: see ``metaharness/models.toml``
(``[system2.astra] backend = "codex", model = "gpt-6-astra"``). This class only
knows how to drive the CLI.

Mirrors Auto-CTFer's codex backend (engine/src/solvers/codex.ts): the CLI's own
login (``~/.codex/auth.json``) is used, ``OPENAI_API_KEY``/``OPENAI_BASE_URL``
are dropped from the child env so the CLI never silently switches auth, and the
reasoning effort is passed as ``model_reasoning_effort``.

``codex exec --json -`` reads the prompt from stdin and prints JSONL events; the final
agent message is also written with ``-o``. ``codex exec resume <thread-id> -`` continues a
thread so Astra's conversation context persists across game turns. This does not establish
persistence of inaccessible model-internal reasoning state.

A profile may name a *provider* (``[providers.<id>]`` in models.toml): an OpenAI-compatible
endpoint such as the local model router Claude Code is configured with, reached with an API key
read from the environment variable the provider names (``.env``). The CLI then runs with
``model_provider`` / ``model_providers.<id>.*`` overrides and its own ``CODEX_HOME`` (sessions,
state; no login, plugins or hooks of the user's codex), and an agentic run's shell never sees the
key (``shell_environment_policy.exclude``).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ._cli import CLITimeout, run_cli
from .base import (
    CLI_ENV_DROP,
    PLAYER_PREAMBLE,
    ModelCall,
    ModelTimeout,
    ModelUnavailable,
    System2Result,
    TraceSink,
    image_trace,
    read_images,
)

# Matched only against what codex reported as errors (error / turn.failed events, stderr of a failed
# process), never the answer: a reply that mentions "usage limit" or a timestamp holding 401 is not one.
# Workspace credits and spend caps are usage limits too (0.160: "Your workspace is out of credits. Add
# credits to continue.", "You hit your spend cap set in your workspace. ...").
_USAGE_LIMIT_RE = re.compile(r"usage limit|insufficient_quota|purchase more credits|exceeded your current quota|"
                             r"quota exceeded\. check your plan|upgrade to (?:plus|pro)\b|out of credits|spend cap",
                             re.I)
_AUTH_RE = re.compile(r"\b401\b|unauthorized|refresh token|sign in again|log ?in required|not logged in|"
                      r"not signed in|authentication required", re.I)
# Tools an inference-only player must not have even if the user's config enables them: view_image
# reads image files by path, plugins bring MCP servers / skills, hooks run per session (and may add
# context), memories are external context. Attached screenshots (-i) are input, not view_image.
_PLAYER_OFF_FEATURES = ("view_image", "image_generation", "plugins", "hooks", "memories")


DEFAULT_CODEX_HOME = "~/.cache/metaharness/codex-home"
_PROVIDER_KEYS = ("name", "base_url", "env_key", "wire_api", "query_params", "http_headers",
                  "env_http_headers", "request_max_retries", "stream_max_retries", "stream_idle_timeout_ms",
                  "requires_openai_auth")


def _toml(value: Any) -> str:
    """A ``-c key=value`` right-hand side: codex parses it as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + "}"
    return json.dumps(str(value))


class CodexCLI:
    def __init__(
        self,
        model: str,
        reasoning_effort: Optional[str] = None,
        codex_bin: str = "codex",
        ephemeral: bool = False,
        extra_config: Optional[dict[str, str]] = None,
        provider: Optional[dict[str, Any]] = None,
    ) -> None:
        """``provider``: ``{"id": ..., "base_url": ..., "env_key": ..., "wire_api": ..., "codex_home": ...}``
        — an OpenAI-compatible endpoint instead of the CLI's ChatGPT login (see the module doc). The
        key must be in the environment (``.env``) when the model is built."""
        self.model = model
        self.name = f"codex:{model}"
        self.reasoning_effort = reasoning_effort
        self.codex_bin = codex_bin
        self.ephemeral = ephemeral
        self.extra_config = dict(extra_config or {})
        self.provider = dict(provider) if provider else None
        self.codex_home: Optional[Path] = None
        if self.provider:
            pid = str(self.provider.get("id") or "router")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
                raise ValueError(f"provider id {pid!r} must be [A-Za-z0-9_-]+")
            if not self.provider.get("base_url"):
                raise ValueError(f"provider {pid!r} needs a base_url")
            self.provider["id"] = pid
            key_var = self.provider.get("env_key")
            if key_var and not os.environ.get(key_var):
                raise ValueError(f"provider {pid!r} reads its API key from ${key_var}, which is not set "
                                 f"(put it in .env, see .env.example)")
            self.codex_home = Path(os.path.expanduser(str(self.provider.get("codex_home") or DEFAULT_CODEX_HOME)))
            self.name = f"codex:{model}@{pid}"
        self.n_calls = 0
        self.total_usage: dict[str, int] = {}     # sum of per-call deltas
        self.player_tools = "disabled-and-audited"
        # turn.completed reports the thread's running total; the last total seen per thread turns
        # it into per-call deltas (a resumed thread would otherwise be charged for every earlier turn).
        self._thread_usage: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    # -- argv -----------------------------------------------------------
    def _argv(self, *, cwd: Optional[Path], session_id: Optional[str],
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
        if self.provider:
            pid = self.provider["id"]
            cfg["model_provider"] = _toml(pid)
            for k in _PROVIDER_KEYS:
                if k in self.provider:
                    cfg[f"model_providers.{pid}.{k}"] = _toml(self.provider[k])
            # the key is for codex, not for the agent's shell (the proposer must not call backends itself)
            if agentic and self.provider.get("env_key"):
                cfg["shell_environment_policy.exclude"] = _toml([self.provider["env_key"]])
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
            for feature in _PLAYER_OFF_FEATURES:
                cfg[f"features.{feature}"] = "false"
            cfg["web_search"] = '"disabled"'
        if self.reasoning_effort:
            cfg["model_reasoning_effort"] = f'"{self.reasoning_effort}"'
        # Silence hooks/notifiers from the user's interactive config.
        cfg.setdefault("notify", "[]")
        for k, v in cfg.items():
            argv += ["-c", f"{k}={v}"]
        # The prompt goes through stdin: one argv element is capped at 128 KiB on Linux (E2BIG) and
        # archived ARC prompts reached 310 KB. Never also a positional prompt: codex would then
        # append stdin to it as a <stdin> block.
        argv.append("-")
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
        install packages / start game servers; ``web_search`` enables live search.
        Running out of time raises ``ModelTimeout`` whose ``session_id`` is the thread the
        call started or resumed, so a caller can resume it."""
        full_prompt = f"{system}\n\n{prompt}" if system and not session_id else prompt
        if not agentic:
            full_prompt = PLAYER_PREAMBLE + "\n\n" + full_prompt
        cwd = Path(cwd).resolve() if cwd else None      # -C and subprocess cwd must both be absolute
        t0 = time.time()
        request: dict[str, Any] = {"prompt": full_prompt}
        try:
            loaded = read_images(images)
        except RuntimeError as e:
            self._trace(trace, t0, {**request, "images": [str(i) for i in images or []]}, None, str(e))
            raise
        if loaded:
            request["images"] = image_trace(loaded)
        with self._lock:     # the running total this thread had reached; None = a thread this client never saw
            known = self._thread_usage.get(session_id) if session_id else {}
            baseline = None if known is None else dict(known)
        with tempfile.TemporaryDirectory() as td:
            last_msg = Path(td) / "last.txt"
            argv = self._argv(cwd=cwd, session_id=session_id, agentic=agentic, last_msg_path=last_msg,
                              images=[str(p) for p, _ in loaded], full_access=full_access, web_search=web_search)
            try:
                proc = run_cli(argv, cwd=cwd, stdin_text=full_prompt, timeout_s=timeout_s, env_drop=CLI_ENV_DROP,
                               env_overrides=self._env_overrides())
            except CLITimeout as e:
                # The partial JSONL still names the thread, so the caller can resume it (repair round).
                thread_id = _digest_events(_parse_jsonl(e.stdout))[0] or session_id
                err = f"timeout: codex exec exceeded {timeout_s:g}s (session {thread_id})"
                self._trace(trace, t0, request, None, err)
                raise ModelTimeout(err, session_id=thread_id) from e
            except OSError as e:    # binary missing, E2BIG, ...: a failed call like any other, not a bare OSError
                err = f"codex exec could not run: {type(e).__name__}: {e}"
                self._trace(trace, t0, request, None, err)
                if _cannot_start(e, self.codex_bin):     # not on PATH / not executable: no call can ever work
                    raise ModelUnavailable(err) from e
                raise RuntimeError(err) from e
            except Exception as e:  # cancelled with its episode, ...
                self._trace(trace, t0, request, None, f"{type(e).__name__}: {e}")
                raise
            text = last_msg.read_text() if last_msg.exists() else ""
        events = _parse_jsonl(proc.stdout)
        thread_id, cumulative, errors = _digest_events(events)
        if not agentic:
            external = [ev.get("item", {}).get("type") for ev in events
                        if ev.get("type") in ("item.started", "item.completed")
                        and ev.get("item", {}).get("type") in
                        {"command_execution", "mcp_tool_call", "web_search", "file_change",
                         "collab_tool_call"}]
            if external:
                error = "inference-only player used external tools: " + ", ".join(sorted(set(external)))
                self._trace(trace, t0, request, None, error)
                raise ModelUnavailable(error)
        turn_failed = any(ev.get("type") == "turn.failed" for ev in events)
        if not text:  # fall back to the last agent_message event
            text = _last_agent_message(events)
        if proc.returncode != 0 or turn_failed or not text.strip():
            # Success is structural (rc 0, no turn.failed, a final message); a failure is classified
            # only from codex's own error reports (+ stderr of a failed process).
            reported = errors + ([(proc.stderr or "")[-2000:]] if proc.returncode != 0 else [])
            detail = "\n".join(m.strip() for m in reported if m.strip()) or (
                "no error output" if proc.returncode != 0 or turn_failed else "empty completed response")
            self._trace(trace, t0, request, None, detail)
            for regex, what in ((_USAGE_LIMIT_RE, "usage limit"), (_AUTH_RE, "auth problem")):
                for msg in reported:
                    hit = regex.search(msg)
                    if hit:
                        raise ModelUnavailable(f"codex {what} (matched {hit.group(0)!r}): {msg.strip()[:300]}")
            if proc.returncode == 0 and not turn_failed:
                raise RuntimeError(f"codex returned no final response: {detail[:800]}")
            raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {detail[:800]}")
        sid = thread_id or session_id
        usage = self._account(sid, session_id, baseline, cumulative)
        latency = time.time() - t0
        res = System2Result(text=text, session_id=sid, latency_s=latency, usage=usage, raw=events)
        # error events inside a turn that still completed (e.g. a retried stream) are warnings
        self._trace(trace, t0, request, res, None, warnings=errors)
        return res

    def _env_overrides(self) -> Optional[dict[str, str]]:
        """A provider run uses its own CODEX_HOME: the user's login, plugins, hooks and MCP servers stay
        out of the players and proposers, and a dead ChatGPT login cannot fail a call. The directory
        gets a config.toml naming the provider, so ``CODEX_HOME=<dir> codex`` reproduces a call by hand."""
        if not self.provider or self.codex_home is None:
            return None
        home = self.codex_home
        with self._lock:
            if not (home / "config.toml").exists():
                home.mkdir(parents=True, exist_ok=True)
                pid = self.provider["id"]
                lines = [f"model = {_toml(self.model)}", f"model_provider = {_toml(pid)}", "",
                         f"[model_providers.{pid}]"]
                lines += [f"{k} = {_toml(self.provider[k])}" for k in _PROVIDER_KEYS if k in self.provider]
                (home / "config.toml").write_text("\n".join(lines) + "\n")
        return {"CODEX_HOME": str(home)}

    def _account(self, thread_id: Optional[str], session_id: Optional[str],
                 baseline: Optional[dict[str, int]], cumulative: dict[str, Any]) -> dict[str, Any]:
        """Per-call usage from the thread's running total; the raw total stays under ``thread_cumulative``."""
        totals = {k: v for k, v in cumulative.items() if isinstance(v, int) and not isinstance(v, bool)}
        base = baseline or {}
        reset = any(v < base.get(k, 0) for k, v in totals.items())   # a total never shrinks: count from zero
        delta = {k: v if reset else v - base.get(k, 0) for k, v in totals.items()}
        with self._lock:
            self.n_calls += 1
            if totals:        # a call that reported nothing keeps the last known total
                for key in {thread_id, session_id} - {None}:
                    self._thread_usage[key] = totals
            for k, v in delta.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v
        if not cumulative:
            return {}
        usage: dict[str, Any] = {**delta, "thread_cumulative": dict(cumulative)}
        if baseline is None:
            usage["thread_baseline"] = "unknown"   # resumed a thread this client never saw: an upper bound
        elif reset:
            usage["thread_baseline"] = "reset"
        return usage

    def _trace(self, trace: Optional[TraceSink], t0: float, request: dict[str, Any], res: Optional[System2Result],
               err: Optional[str], warnings: Optional[list[str]] = None) -> None:
        if not trace:
            return
        response: Optional[dict[str, Any]] = None
        if res:
            response = {"text": res.text, "session_id": res.session_id}
            if warnings:
                response["warnings"] = list(warnings)
        trace.log_call(ModelCall(
            "system2", self.name, t0, time.time() - t0, request, response,
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


def _cannot_start(e: OSError, binary: str) -> bool:
    """The CLI executable itself could not be started (Popen sets filename to it; a missing cwd or
    E2BIG are different failures)."""
    import errno
    return (isinstance(e, (FileNotFoundError, PermissionError)) or e.errno == errno.ENOEXEC) and \
        str(e.filename or "") in (binary, str(binary))
