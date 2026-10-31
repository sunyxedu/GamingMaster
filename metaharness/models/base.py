"""Model interfaces.

Two kinds of model, mirroring the fast/slow split of the tier-2 design:

* ``System2Model`` — a slow, reasoning, *agentic* model (Astra through the
  codex CLI). Text in, text out, optionally resuming a session so the model's
  conversation context carries over between calls. It may also be run as a
  coding agent inside a working directory (that is how the meta-harness
  proposer uses it).
* ``System1Model`` — a fast, non-autoregressive decision model (Jev). Typed
  questions in (choice / score / noul), calibrated probabilities out.

Every call is recorded as a ``ModelCall`` and pushed to an optional
``TraceSink`` so harness evaluations leave a full trace on disk.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


class ModelUnavailable(RuntimeError):
    """The backend cannot serve requests right now (quota, auth, network)."""


class ModelCallCancelled(RuntimeError):
    """The episode that made the call ended while it was in flight; the backend process was killed."""


class ModelTimeout(TimeoutError):
    """A model call ran out of time. ``session_id`` is the conversation the call had started or
    resumed (None if unknown), so a caller can resume it — e.g. a repair round for a coding agent
    that timed out during its final checks."""

    def __init__(self, message: str, session_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.session_id = session_id


@dataclass
class ModelCall:
    kind: str                 # "system1" | "system2"
    backend: str
    t_start: float
    latency_s: float
    request: Any
    response: Any
    ok: bool = True
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class TraceSink:
    """Append-only JSONL trace; thread-safe. ``None`` path = in-memory only."""

    def __init__(self, path: Path | str | None = None, *, append: bool = True) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fh = open(self.path, "a" if append else "w", encoding="utf-8") if self.path else None

    def write(self, record: Mapping[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("t", time.time())
        with self._lock:
            self.records.append(rec)
            if self._fh:
                self._fh.write(json.dumps(rec, default=str) + "\n")
                self._fh.flush()

    def log_call(self, call: ModelCall) -> None:
        self.write({
            "type": "model_call", "kind": call.kind, "backend": call.backend,
            "t_start": call.t_start, "latency_s": call.latency_s, "ok": call.ok,
            "error": call.error, "usage": call.usage,
            "request": call.request, "response": call.response,
        })

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ----------------------------------------------------------------- System 2

@dataclass
class System2Result:
    text: str
    session_id: str | None = None
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class System2Model(Protocol):
    name: str

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
        """One turn. ``session_id`` resumes a prior turn (conversation context kept).

        ``agentic=True`` lets the model use tools (shell / file edits) inside
        ``cwd`` — used by the meta-harness proposer. ``agentic=False`` asks for
        a plain answer (still through the same CLI so the model is the same).
        """
        ...


# Sent with every agentic=False call by the CLI backends (codex: prepended to the prompt; claude:
# its system prompt). A player is inference-only: files, search or memories could expose
# simulator state or answers outside its observation.
PLAYER_PREAMBLE = ("Inference-only game player: use only the supplied observations and "
                   "conversation. Do not use tools, read files, run commands, browse, or "
                   "consult external memories. Return the requested answer directly.")

# Put before an agent's own system prompt (it replaces the CLI's default, which asks for parallel tool calls):
# agents that kept the default batched 1.7 tool calls per turn, the others 1.2.
AGENT_PREAMBLE = ("Each of your turns is one slow model call (often a minute under a shared rate limit): do as much "
                  "as you can per turn. Make independent tool calls together in one turn (several reads, searches or "
                  "commands at once), chain shell steps into one command, and write each file in few, complete edits.")

# Never inherited by a CLI child: each CLI uses its own login (OPENAI_* would silently switch codex
# to API-key auth), and an agent's shell has no business with another backend's credentials.
CLI_ENV_DROP = ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY")


def read_images(images: Optional[list[str]]) -> list[tuple[Path, bytes]]:
    """Absolute path + bytes of each image a System-2 call attaches, read up front so the trace
    names exactly what the model saw. An unreadable image is an error, never a silent drop."""
    out: list[tuple[Path, bytes]] = []
    for img in images or []:
        path = Path(img).resolve()
        try:
            out.append((path, path.read_bytes()))
        except OSError as e:
            raise RuntimeError(f"cannot read image {img}: {e}") from e
    return out


def image_trace(images: list[tuple[Path, bytes]]) -> list[dict[str, str]]:
    return [{"path": str(p), "sha256": hashlib.sha256(data).hexdigest()} for p, data in images]


# ----------------------------------------------------------------- System 1

@dataclass
class System1Answer:
    """One answer of a System-1 request (normalised across backends)."""
    type: str                                   # choice | score | noul
    value: Any                                  # chosen option / score / p(yes)
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]


class System1Model(Protocol):
    name: str

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        timeout_s: float = 5.0,
        trace: Optional[TraceSink] = None,
    ) -> dict[str, System1Answer]:
        ...
