"""Offline stand-ins used by tests and by ``--system1 heuristic`` /
``--system2 scripted`` dry runs. They report *declared* latencies so the
virtual-clock runner can charge realistic time without network."""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional

from .base import ModelCall, ModelTimeout, System1Answer, System2Result, TraceSink


class ScriptedSystem2:
    """Returns canned replies in order (last one repeats), or calls ``fn(prompt)``."""
    name = "scripted"

    def __init__(self, replies: Optional[list[str]] = None, fn: Optional[Callable[[str], str]] = None,
                 latency_s: float = 1.5, files: Optional[list[dict[str, str]]] = None) -> None:
        self.replies = list(replies or ["OK"])
        self.fn = fn
        # ``files``: per call, {relative_path: content} written into ``cwd`` when
        # run agentically (a stand-in for a coding agent that edits the workspace).
        self.files = list(files or [])
        self.latency_s = latency_s
        self.calls: list[str] = []
        self.n_calls = 0

    def run(self, prompt: str, *, cwd=None, session_id=None, system=None, agentic=False,
            timeout_s: float = 600.0, trace: Optional[TraceSink] = None, images=None,
            full_access: bool = False, web_search: bool = False) -> System2Result:
        self.calls.append(prompt)
        if self.latency_s > timeout_s:
            # A real backend would time out here; pre-flight must exercise the harness's timeout path.
            err = f"timeout: scripted latency {self.latency_s:g}s exceeds the call's {timeout_s:g}s"
            if trace:
                trace.log_call(ModelCall("system2", "scripted", time.time(), timeout_s, {"prompt": prompt}, None,
                                         ok=False, error=err))
            raise ModelTimeout(err, session_id=session_id or "scripted-session")
        if agentic and cwd is not None and self.files:
            from pathlib import Path
            batch = self.files[min(len(self.calls) - 1, len(self.files) - 1)]
            for rel, content in batch.items():
                target = Path(cwd) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        text = self.fn(prompt) if self.fn else self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        self.n_calls += 1
        res = System2Result(text=text, session_id=session_id or "scripted-session", latency_s=self.latency_s)
        if trace:
            trace.log_call(ModelCall("system2", "scripted", time.time(), self.latency_s, {"prompt": prompt}, {"text": text}))
        return res


class HeuristicSystem1:
    """Answers questions with a user-supplied policy ``fn(state, questions) -> {qid: option}``.

    Without ``fn`` it picks the first option of every choice (a deliberately
    dumb baseline so that a harness cannot look good by accident).
    """
    name = "heuristic"

    def __init__(self, fn: Optional[Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]] = None,
                 latency_s: float = 0.3) -> None:
        self.fn = fn
        self.latency_s = latency_s
        self.n_calls = 0
        self.total_input_tokens = 0

    def decide(self, state: Any, questions: Mapping[str, Any], *, timeout_s: float = 5.0,
               trace: Optional[TraceSink] = None) -> dict[str, System1Answer]:
        if self.latency_s > timeout_s:      # as a real backend would; exercises harness timeout paths
            err = f"timeout: heuristic latency {self.latency_s:g}s exceeds the call's {timeout_s:g}s"
            if trace:
                trace.log_call(ModelCall("system1", "heuristic", time.time(), timeout_s, {"state": state, "questions": {
                    k: (v.to_json() if hasattr(v, 'to_json') else v) for k, v in questions.items()}}, None,
                    ok=False, error=err))
            raise TimeoutError(err)
        picks = dict(self.fn(state, questions)) if self.fn else {}
        out: dict[str, System1Answer] = {}
        for qid, q in questions.items():
            qj = q.to_json() if hasattr(q, "to_json") else dict(q)
            t = qj.get("type")
            if t == "choice":
                opts = list(qj["criteria"].keys())
                pick = picks.get(qid, opts[0])
                probs = {o: (0.7 if o == pick else 0.3 / max(1, len(opts) - 1)) for o in opts}
                out[qid] = System1Answer("choice", pick, probs, 0.7)
            elif t == "score":
                levels = list(qj["criteria"])
                pick = picks.get(qid, 0)
                probs = {str(i): (0.7 if i == pick else 0.3 / max(1, len(levels) - 1)) for i in range(len(levels))}
                out[qid] = System1Answer("score", float(pick), probs, 0.7)
            else:
                p = float(picks.get(qid, 0.5))
                out[qid] = System1Answer("noul", p, {"true": p, "false": 1 - p}, None)
        self.n_calls += 1
        if trace:
            trace.log_call(ModelCall("system1", "heuristic", time.time(), self.latency_s,
                                     {"state": state, "questions": {k: (v.to_json() if hasattr(v, 'to_json') else v) for k, v in questions.items()}},
                                     {k: {"value": a.value, "probabilities": a.probabilities} for k, a in out.items()}))
        return out


class SilentModel:
    """Every call fails as a backend error would (RuntimeError): the no-models ablation. A harness still
    sees ``ctx.system1`` / ``ctx.system2`` and plays on its own fallbacks."""
    name = "silent"
    model = None

    def _fail(self, kind: str, trace: Optional[TraceSink]) -> None:
        err = "no answer: this evaluation runs without runtime models"
        if trace:
            trace.log_call(ModelCall(kind, "silent", time.time(), 0.0, {}, None, ok=False, error=err))
        raise RuntimeError(err)

    def decide(self, state: Any, questions: Mapping[str, Any], *, timeout_s: float = 5.0,
               trace: Optional[TraceSink] = None) -> dict[str, System1Answer]:
        self._fail("system1", trace)
        return {}

    def run(self, prompt: str, *, trace: Optional[TraceSink] = None, **kw: Any) -> System2Result:
        self._fail("system2", trace)
        return System2Result(text="")
