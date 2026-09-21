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

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


class ModelUnavailable(RuntimeError):
    """The backend cannot serve requests right now (quota, auth, network)."""


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
