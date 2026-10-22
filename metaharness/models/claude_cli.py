"""Any Claude model through the ``claude`` CLI (``claude -p``).

Which model is a config concern (``metaharness/models.toml``); this class only
drives the CLI. Same interface and session semantics (``--resume``) as the
codex backend. Never substituted silently for another profile.

The prompt goes through stdin. Text-only calls use ``--output-format json`` (one result
object). Calls with images use ``--input-format stream-json --output-format stream-json``:
stdin is one user message whose content is the base64 images followed by the text, and the
answer is the stream's final ``result`` event (same fields as the json output).

A player (``agentic=False``) is isolated from the user's Claude Code setup: no tools, MCP
servers, hooks, skills / slash commands, CLAUDE.md or auto-memory, the inference-only
instruction as its system prompt (instead of the coding-agent prompt with env and git status),
and a private empty working directory, created once per instance and reused because sessions
are stored per directory. Not ``--bare``: it also drops settings-file auth
(ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL in settings env), which some installs rely on.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
import weakref
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

# `claude --help` (2.1.308); the CLI only warns on an unknown --effort and runs at its default.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Quota / auth messages of the CLI and the API. Checked only on a failed call (rc != 0 or is_error);
# "overloaded", API 5xx and timeouts are transient (RuntimeError / TimeoutError).
_UNAVAILABLE_RE = re.compile(
    r"hit your|reached your|usage limit|session limit|weekly limit|out of extra usage|"
    r"doesn.t include extra usage|requires usage credits|credit balance|run /login|sign in again|"
    r"invalid api key|invalid auth token|oauth token|authentication required|not logged in", re.I)
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp"}
_ENV_DROP = CLI_ENV_DROP + ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")   # a nested session refuses to start
_PLAYER_ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
_PLAYER_SETTINGS = json.dumps({"disableAllHooks": True})


class ClaudeCLI:
    def __init__(self, model: Optional[str] = None, reasoning_effort: Optional[str] = None,
                 claude_bin: str = "claude") -> None:
        if reasoning_effort and reasoning_effort not in EFFORT_LEVELS:
            raise ValueError(f"claude reasoning_effort must be one of {', '.join(EFFORT_LEVELS)}, "
                             f"got {reasoning_effort!r}")
        self.model = model
        self.name = f"claude:{model or 'default'}"
        self.reasoning_effort = reasoning_effort or None      # passed as --effort on every call
        self.claude_bin = claude_bin
        self.n_calls = 0
        self.total_cost_usd = 0.0
        self.player_tools = "disabled"
        self._lock = threading.Lock()
        self._player_dir: Optional[Path] = None

    def _player_cwd(self) -> Path:
        """Empty directory the players run in: nothing project-specific to discover, and the same
        directory every call so --resume finds the sessions Claude Code stores per directory."""
        with self._lock:
            if self._player_dir is None:
                self._player_dir = Path(tempfile.mkdtemp(prefix="metaharness-claude-player-")).resolve()
                weakref.finalize(self, shutil.rmtree, str(self._player_dir), ignore_errors=True)
            self._player_dir.mkdir(parents=True, exist_ok=True)   # same path if a tmp cleaner removed it
            return self._player_dir

    def _argv(self, *, session_id: Optional[str], new_session_id: Optional[str], system: Optional[str],
              agentic: bool, web_search: bool, stream: bool) -> list[str]:
        argv = [self.claude_bin, "-p"]
        if stream:      # stream-json input requires stream-json output, which requires --verbose
            argv += ["--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
        else:
            argv += ["--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]
        if self.reasoning_effort:
            argv += ["--effort", self.reasoning_effort]
        if session_id:
            argv += ["--resume", session_id]
        elif new_session_id:    # chosen up front, so a call that times out can still be resumed
            argv += ["--session-id", new_session_id]
        if agentic:
            if system and not session_id:
                argv += ["--system-prompt", system]
            argv += ["--permission-mode", "bypassPermissions", "--dangerously-skip-permissions"]
            if not web_search:  # study / adapt research the game on the web; the proposer must not
                argv += ["--disallowedTools", "WebSearch", "WebFetch"]
        else:
            # Replaces the coding-agent system prompt. Sent on --resume too: the CLI keeps reusing
            # the prompt it recorded first, and renders this one again after a compaction.
            argv += ["--system-prompt", PLAYER_PREAMBLE + (f"\n\n{system}" if system else ""),
                     "--tools", "", "--strict-mcp-config", "--settings", _PLAYER_SETTINGS,
                     "--disable-slash-commands"]
        return argv

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
        """``full_access`` needs nothing extra (agentic calls already bypass permissions).
        Running out of time raises ``ModelTimeout`` naming the session it started or resumed."""
        t0 = time.time()
        request: dict[str, Any] = {"prompt": prompt}
        if not agentic:
            request["system"] = PLAYER_PREAMBLE + (f"\n\n{system}" if system else "")
        elif system and not session_id:
            request["system"] = system
        try:
            loaded = read_images(images)
            blocks = [_image_block(path, data) for path, data in loaded]
        except RuntimeError as e:
            self._trace(trace, t0, {**request, "images": [str(i) for i in images or []]}, None, str(e))
            raise
        if loaded:
            request["images"] = image_trace(loaded)
        stream = bool(blocks)
        new_session_id = None if session_id else str(uuid.uuid4())
        sid = session_id or new_session_id
        argv = self._argv(session_id=session_id, new_session_id=new_session_id, system=system,
                          agentic=agentic, web_search=web_search, stream=stream)
        stdin = prompt if not stream else json.dumps(
            {"type": "user", "message": {"role": "user", "content": blocks + [{"type": "text", "text": prompt}]}}) + "\n"
        run_dir = (Path(cwd).resolve() if cwd else None) if agentic else self._player_cwd()
        try:
            proc = run_cli(argv, cwd=run_dir, stdin_text=stdin, timeout_s=timeout_s,
                           env_overrides=None if agentic else _PLAYER_ENV, env_drop=_ENV_DROP)
        except CLITimeout as e:
            err = f"timeout: claude -p exceeded {timeout_s:g}s (session {sid})"
            self._trace(trace, t0, request, None, err)
            raise ModelTimeout(err, session_id=sid) from e
        except Exception as e:  # binary missing, cancelled with its episode, ...
            self._trace(trace, t0, request, None, f"{type(e).__name__}: {e}")
            raise
        latency = time.time() - t0
        payload = _result_payload(proc.stdout or "", stream)
        if proc.returncode != 0 or (payload is not None and payload.get("is_error")):
            err = _error_text(payload, proc)
            self._trace(trace, t0, request, None, err)
            hit = _UNAVAILABLE_RE.search(err)
            if hit:
                raise ModelUnavailable(f"claude unavailable (matched {hit.group(0)!r}): {err[-800:]}")
            raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {err}")
        text = payload.get("result") if payload is not None else None
        if not isinstance(text, str) or not text.strip():
            err = ("claude -p printed no JSON result: " + repr((proc.stdout or "")[-500:]) if payload is None
                   else "claude -p returned an empty result")
            self._trace(trace, t0, request, None, err)
            raise RuntimeError(err)
        raw_usage, cost = payload.get("usage"), payload.get("total_cost_usd")
        usage = {"total_cost_usd": cost, **(raw_usage if isinstance(raw_usage, dict) else {})}
        with self._lock:
            self.n_calls += 1
            self.total_cost_usd += float(cost) if isinstance(cost, (int, float)) else 0.0
        res = System2Result(text=text, session_id=payload.get("session_id") or sid, latency_s=latency,
                            usage=usage, raw=payload)
        self._trace(trace, t0, request, res, None)
        return res

    def _trace(self, trace: Optional[TraceSink], t0: float, request: dict[str, Any],
               res: Optional[System2Result], err: Optional[str]) -> None:
        if trace:
            trace.log_call(ModelCall(
                "system2", self.name, t0, time.time() - t0, request,
                {"text": res.text, "session_id": res.session_id} if res else None,
                ok=err is None, error=err, usage=res.usage if res else {}))


def _image_block(path: Path, data: bytes) -> dict[str, Any]:
    if not data:
        raise RuntimeError(f"image {path} is empty")
    media_type = _sniff_media_type(data) or _MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        raise RuntimeError(f"image {path}: not png / jpeg / gif / webp")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                       "data": base64.b64encode(data).decode("ascii")}}


def _sniff_media_type(data: bytes) -> Optional[str]:
    """The bytes win over the suffix: the API rejects an image whose declared type does not match."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _result_payload(stdout: str, stream: bool) -> Optional[dict[str, Any]]:
    """The final ``result`` object: of a stream, or of json output (one object; a JSON array of
    every message when verbose is configured). Lines that are not JSON are skipped."""
    found: list[Any] = []
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        found.extend(obj if isinstance(obj, list) else [obj])
    results = [o for o in found if isinstance(o, dict) and o.get("type") == "result"]
    if results:
        return results[-1]
    if not stream and found and isinstance(found[-1], dict):
        return found[-1]          # an untyped json result (older CLIs)
    return None


def _error_text(payload: Optional[dict[str, Any]], proc: Any) -> str:
    """What the CLI reported as the error. ``result`` is an error message only with ``is_error``;
    otherwise it is an answer, which must not be classified (a non-zero exit after it still fails)."""
    parts: list[str] = []
    if payload is not None:
        if payload.get("is_error") and isinstance(payload.get("result"), str) and payload["result"].strip():
            parts.append(payload["result"])
        errors = payload.get("errors")
        for e in errors if isinstance(errors, list) else [errors] if errors else []:
            parts.append(e if isinstance(e, str) else json.dumps(e, default=str))
    if (proc.stderr or "").strip():
        parts.append(proc.stderr.strip()[-1500:])
    if not parts and payload is None and (proc.stdout or "").strip():
        parts.append(proc.stdout.strip()[-1500:])
    return "\n".join(parts)[-1500:] or f"exit status {proc.returncode}, no error output"
