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
are stored per directory. Agents (study / adapt / proposer) get the file, shell and (when asked)
web tools only — no subagents, skills, MCP servers or cross-session tools, which multiplied
requests against a shared rate limit and let an agent see other sessions — and the same CLAUDE.md,
auto-memory and hook isolation: their run directory is their whole world. Not ``--bare``: it also drops
settings-file auth (ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL in settings env), which some
installs rely on.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any, Iterator, Optional

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
# Quota / auth messages of the CLI (its usage-limit and auth texts, checked in the 2.1.309 binary) and
# the API. Checked only on a failed call (rc != 0 or is_error); "overloaded", API 5xx and timeouts are
# transient (RuntimeError / TimeoutError). "Usage credits are required for long context" is not here:
# a shorter request still gets through.
_UNAVAILABLE_RE = re.compile(
    r"hit your|reached your|usage limit|session limit|weekly limit|out of (?:extra )?usage|"
    r"doesn.t include (?:extra )?usage|requires usage credits|usage allocation has been disabled|"
    r"disabled for your org|disabled organization|organization has disabled|credit balance|run /login|"
    r"sign in again|login expired|invalid api key|invalid auth token|oauth token|authentication required|"
    r"not logged in", re.I)
# Lines the CLI itself calls temporary though they name a limit or a login: a 429 that is "not your
# usage limit", an OAuth refresh another claude process holds ("retry in a minute"). Not classified.
_TRANSIENT_LINE_RE = re.compile(r"^.*(?:not your usage limit|another Claude Code process is refreshing).*$",
                                re.I | re.M)
# One argv element is capped at 128 KiB on Linux (E2BIG): a longer system prompt goes through a file.
_SYSTEM_ARGV_MAX = 32 * 1024
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp"}
_ENV_DROP = CLI_ENV_DROP + ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")   # a nested session refuses to start
# A requests-per-minute cap at the API (a shared router enforces one per user) is hit by every process on
# the machine together. Two defences: a cross-process cap on concurrent invocations — agents and players
# have separate pools (METAHARNESS_CLAUDE_AGENT_SLOTS / METAHARNESS_CLAUDE_PLAYER_SLOTS; 0 = no cap) —
# and, when the CLI still gives up with 429 after its own retries, our retry with backoff that resumes
# the same session so an agent's work is not lost. 429 is transient, never ModelUnavailable.
_SLOT_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "metaharness" / "claude-slots"
_SLOT_DEFAULTS = {"agent": 4, "player": 8}
# Retried (agents resume their session): rate limits and transient API / stream failures. Quota and auth
# failures (_UNAVAILABLE_RE) never are.
_RETRY_RE = re.compile(r"\b429\b|rate.?limit|too many requests|overloaded|\b5\d\d\b|upstream|stream error|"
                       r"internal server error|service unavailable|bad gateway|gateway time-?out|temporar|"
                       r"econnreset|socket hang up|connection (?:reset|refused|error|closed)|request timed out", re.I)
_RETRY_WAITS = (15.0, 30.0, 60.0, 120.0, 240.0)
_sleep = time.sleep          # module level, so tests can skip the waits
# a shell command may run up to 15 min by default (a harness try, a build), 30 min when the agent asks
_ISOLATION_ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                  "BASH_DEFAULT_TIMEOUT_MS": "900000", "BASH_MAX_TIMEOUT_MS": "1800000"}
_AGENT_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit")
_WEB_TOOLS = ("WebSearch", "WebFetch")


def _isolation_settings() -> str:
    """``--settings`` for every call: no hooks; and, with METAHARNESS_CLAUDE_BASE_URL set, the API base URL
    (a local queueing proxy in front of a shared router) — the flag outranks the user's settings env."""
    settings: dict[str, Any] = {"disableAllHooks": True}
    base = os.environ.get("METAHARNESS_CLAUDE_BASE_URL")
    if base:
        settings["env"] = {"ANTHROPIC_BASE_URL": base}
    return json.dumps(settings)


@contextlib.contextmanager
def _slot(pool: str) -> Iterator[None]:
    """Hold one of the pool's slots (lock files shared by every process of this user) for the call."""
    n = int(os.environ.get(f"METAHARNESS_CLAUDE_{pool.upper()}_SLOTS", _SLOT_DEFAULTS[pool]) or 0)
    try:
        import fcntl
    except ImportError:          # not POSIX: no cross-process cap
        n = 0
    if n <= 0:
        yield
        return
    _SLOT_DIR.mkdir(parents=True, exist_ok=True)
    handles = [open(_SLOT_DIR / f"{pool}{i}.lock", "a") for i in range(n)]
    held = None
    try:
        while held is None:
            for fh in handles:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = fh
                    break
                except OSError:
                    continue
            if held is None:
                time.sleep(0.1 + random.random() * 0.2)
        yield
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
        for fh in handles:
            fh.close()


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
              agentic: bool, web_search: bool, stream: bool, system_file: Optional[Path] = None) -> list[str]:
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
                argv += _system_flag(system, system_file)
            # study / adapt research the game on the web; the proposer must not
            tools = _AGENT_TOOLS + (_WEB_TOOLS if web_search else ())
            argv += ["--tools", ",".join(tools), "--strict-mcp-config", "--disable-slash-commands",
                     "--permission-mode", "bypassPermissions", "--dangerously-skip-permissions",
                     "--settings", _isolation_settings()]
            if not web_search:
                argv += ["--disallowedTools", *_WEB_TOOLS]
        else:
            # Replaces the coding-agent system prompt. Sent on --resume too: the CLI keeps reusing
            # the prompt it recorded first, and renders this one again after a compaction.
            argv += _system_flag(PLAYER_PREAMBLE + (f"\n\n{system}" if system else ""), system_file)
            argv += ["--tools", "", "--strict-mcp-config", "--settings", _isolation_settings(),
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
        stdin = prompt if not stream else json.dumps(
            {"type": "user", "message": {"role": "user", "content": blocks + [{"type": "text", "text": prompt}]}}) + "\n"
        system_file: Optional[Path] = None
        deadline = time.monotonic() + timeout_s
        attempt = 0
        toolless_retry = False
        working = False          # an agent's time starts with its slot: queueing is not working time
        try:
            run_dir = (Path(cwd).resolve() if cwd else None) if agentic else self._player_cwd()
            if len(request.get("system", "").encode("utf-8")) > _SYSTEM_ARGV_MAX:
                # a player's in its private directory; an agent's in the temp dir, never its workspace
                system_file = _system_file(request["system"], None if agentic else run_dir)
            while True:
                argv = self._argv(session_id=session_id, new_session_id=new_session_id, system=system,
                                  agentic=agentic, web_search=web_search, stream=stream, system_file=system_file)
                with _slot("agent" if agentic else "player"):
                    if agentic and not working:
                        deadline, working = time.monotonic() + timeout_s, True
                    if deadline - time.monotonic() <= 0:     # a player whose time ran out waiting for a slot
                        raise CLITimeout(f"no time left after waiting {timeout_s:g}s for a free slot")
                    proc = run_cli(argv, cwd=run_dir, stdin_text=stdin, timeout_s=max(1.0, deadline - time.monotonic()),
                                   env_overrides=_ISOLATION_ENV, env_drop=_ENV_DROP)
                payload = _result_payload(proc.stdout or "", stream)
                failed = proc.returncode != 0 or (payload is not None and payload.get("is_error"))
                err = _error_text(payload, proc) if failed else ""
                if not failed and agentic and new_session_id and not toolless_retry and payload is not None \
                        and payload.get("num_turns") == 1:
                    # a new agent session that ended without one tool call: its tools did not reach the model
                    # (seen behind the router); resuming would keep its "I cannot access files" belief
                    toolless_retry = True
                    self._trace(trace, t0, request, None, "the agent ended without using any tool; retrying in a new session")
                    new_session_id = sid = str(uuid.uuid4())
                    continue
                if not failed or attempt >= len(_RETRY_WAITS) or not _RETRY_RE.search(err) \
                        or _UNAVAILABLE_RE.search(_TRANSIENT_LINE_RE.sub("", err)):
                    break
                wait = _RETRY_WAITS[attempt] * (0.75 + random.random() * 0.5)
                if time.monotonic() + wait >= deadline:
                    break
                attempt += 1
                self._trace(trace, t0, request, None, f"transient failure (retry {attempt} in {wait:.0f}s): {err[-300:]}")
                _sleep(wait)
                # an agent resumes its session (its work so far is kept); a player repeats the request
                if agentic and _session_exists(run_dir, sid):
                    session_id, new_session_id = sid, None
                    stdin = "The previous turn was interrupted by a transient API error. Continue where you left off."
                elif not session_id:
                    new_session_id = sid = str(uuid.uuid4())
        except CLITimeout as e:
            err = f"timeout: claude -p exceeded {timeout_s:g}s (session {sid})"
            self._trace(trace, t0, request, None, err)
            raise ModelTimeout(err, session_id=sid) from e
        except OSError as e:    # binary missing, E2BIG, ...: a failed call like any other, not a bare OSError
            err = f"claude -p could not run: {type(e).__name__}: {e}"
            self._trace(trace, t0, request, None, err)
            if _cannot_start(e, self.claude_bin):     # not on PATH / not executable: no call can ever work
                raise ModelUnavailable(err) from e
            raise RuntimeError(err) from e
        except Exception as e:  # cancelled with its episode, ...
            self._trace(trace, t0, request, None, f"{type(e).__name__}: {e}")
            raise
        finally:
            if system_file is not None:
                with contextlib.suppress(OSError):
                    system_file.unlink()
        latency = time.time() - t0
        if failed:
            self._trace(trace, t0, request, None, err)
            hit = _UNAVAILABLE_RE.search(_TRANSIENT_LINE_RE.sub("", err))
            if hit:
                raise ModelUnavailable(f"claude unavailable (matched {hit.group(0)!r}): {err[-800:]}")
            if _RETRY_RE.search(err):
                raise RuntimeError(f"claude -p failed after {attempt} retries (transient): {err[-500:]}")
            raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {err}")
        text = payload.get("result") if payload is not None else None
        if agentic and payload is not None and not isinstance(text, str) or agentic and isinstance(text, str) \
                and not text.strip():
            text = ""            # an agent's work is in its files; ending its turn without a final message is fine
        elif not isinstance(text, str) or not text.strip():
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


def _session_exists(run_dir: Optional[Path], sid: Optional[str]) -> bool:
    """Whether the CLI stored a conversation ``sid`` for ``run_dir`` (sessions live per project directory)."""
    if not sid or run_dir is None:
        return False
    projects = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
    return any(p.name == f"{sid}.jsonl" for d in projects.glob("*") if d.is_dir() for p in d.glob(f"{sid}.jsonl"))


def _system_file(text: str, directory: Optional[Path]) -> Path:
    """A private file holding a system prompt too long for argv; the CLI reads it at start-up."""
    fd, path = tempfile.mkstemp(prefix="metaharness-claude-system-", suffix=".txt", dir=directory)
    with open(fd, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return Path(path)


def _system_flag(text: str, path: Optional[Path]) -> list[str]:
    return ["--system-prompt-file", str(path)] if path is not None else ["--system-prompt", text]


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


def _cannot_start(e: OSError, binary: str) -> bool:
    """The CLI executable itself could not be started (Popen sets filename to it; a missing cwd or
    E2BIG are different failures)."""
    import errno
    return (isinstance(e, (FileNotFoundError, PermissionError)) or e.errno == errno.ENOEXEC) and \
        str(e.filename or "") in (binary, str(binary))
