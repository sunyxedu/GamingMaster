"""Jev — TypeSafe's System-One model (https://docs.typesafe.ai/api).

POST https://api.typesafe.ai/v1/systemone
  {"model": "jev-latest", "state": <str|obj|list>, "questions": {id: {...}}}
Question types: ``choice`` (criteria: {option: description}), ``score``
(criteria: [level0, level1, ...]), ``noul`` (yes/no; criteria optional).
Answers carry calibrated probabilities + confidence. Latency 70–700 ms.

Standard library only (urllib) so the fast loop has no extra deps.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..config import load_dotenv
from .base import ModelCall, ModelUnavailable, System1Answer, TraceSink

JEV_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass
class Choice:
    instructions: str
    criteria: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass
class Score:
    instructions: str
    criteria: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass
class Noul:
    instructions: str
    criteria: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = dict(self.criteria)
        return d


def _q_json(q: Any) -> dict[str, Any]:
    if hasattr(q, "to_json"):
        return q.to_json()
    if isinstance(q, Mapping):
        return dict(q)
    raise TypeError(f"question must be Choice/Score/Noul or a dict, got {type(q).__name__}")


def parse_answer(a: Mapping[str, Any]) -> System1Answer:
    t = a.get("type")
    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
    conf = a.get("confidence")
    if t == "choice":
        return System1Answer("choice", a.get("choice"), probs, conf)
    if t == "score":
        return System1Answer("score", a.get("score"), probs, conf)
    if t == "noul":
        p = float(a.get("noul", 0.0))
        return System1Answer("noul", p, {"true": p, "false": 1.0 - p}, conf)
    return System1Answer(str(t), a, probs, conf)


class JevClient:
    name = "jev"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        url: str = JEV_URL,
        max_retries: int = 3,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise ModelUnavailable("TYPESAFE_API_KEY missing — put it in .env")
        self.model = model
        self.url = url
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.n_calls = 0

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        timeout_s: float = 5.0,
        trace: Optional[TraceSink] = None,
    ) -> dict[str, System1Answer]:
        body = {"model": self.model, "state": state, "questions": {k: _q_json(q) for k, q in questions.items()}}
        data = json.dumps(body).encode()
        t0 = time.time()
        last_err: str | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.url, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    payload = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (429, 529) and attempt < self.max_retries:
                    time.sleep(min(2.0, 0.2 * 2**attempt) + random.random() * 0.1)
                    continue
                self._log(trace, t0, body, None, last_err)
                if e.code in (401, 403):
                    raise ModelUnavailable(last_err) from e
                raise RuntimeError(last_err) from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(0.2 * 2**attempt)
                    continue
                self._log(trace, t0, body, None, last_err)
                raise RuntimeError(last_err) from e
        latency = time.time() - t0
        usage = payload.get("usage", {}) or {}
        self.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.n_calls += 1
        if trace:
            trace.log_call(ModelCall("system1", f"jev:{payload.get('model', self.model)}", t0, latency, body, payload, usage=usage))
        return {k: parse_answer(v) for k, v in (payload.get("answers") or {}).items()}

    def _log(self, trace: Optional[TraceSink], t0: float, body: Any, payload: Any, err: str | None) -> None:
        if trace:
            trace.log_call(ModelCall("system1", "jev", t0, time.time() - t0, body, payload, ok=err is None, error=err))
