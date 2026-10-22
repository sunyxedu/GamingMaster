"""Jev — TypeSafe's System-One model (https://docs.typesafe.ai/api).

POST https://api.typesafe.ai/v1/systemone
  {"model": <model id from models.toml>, "state": <str|obj|list>, "questions": {id: {...}}}
Question types (contract per docs.typesafe.ai/api, checked 2026-10-22):
  ``choice``  instructions: str|obj|list; criteria: {option: description|None}   (<=255 options)
  ``score``   instructions: str|obj|list; criteria: [level0, level1, ...]         (2..10 levels)
  ``noul``    instructions: str|obj|list; criteria: {"true": ..., "false": ...} optional
Answers carry calibrated probabilities + confidence (noul: probability only); each answer has its
question's type. Question ids are for code only — the model never sees them, so the whole
meaning must be in ``instructions``/``criteria``. Independent questions over
the same state should be sent in ONE request (evaluated in parallel).

Standard library only (http.client) so the fast loop has no extra deps. When
working on this integration use the ``typesafe-ai`` skill / live docs.
"""
from __future__ import annotations

import base64
import email.utils
import http.client
import json
import math
import os
import random
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from ..config import load_dotenv
from .base import ModelCall, ModelUnavailable, System1Answer, TraceSink

JEV_URL = "https://api.typesafe.ai/v1/systemone"
MAX_CHOICE_OPTIONS = 255     # API: criteria maxProperties
MAX_SCORE_LEVELS = 10        # API: criteria maxItems (and minItems 1; locally >= 2 levels)
# Throttling (429/529) passes; it only makes Jev "unavailable" once it outlasts this many
# consecutive calls of one client spanning at least this long (see JevClient).
THROTTLED_CALLS_UNAVAILABLE = 5
THROTTLED_SPAN_UNAVAILABLE_S = 30.0
_RETRY_STATUS = frozenset({408, 429, *range(500, 600)})
_THROTTLE_STATUS = frozenset({429, 529})
_UNAVAILABLE_STATUS = frozenset({401, 402, 403})
# A 4xx naming billing / quota exhaustion (a 429 included) is not something a retry gets past.
_QUOTA_RE = re.compile(r"billing|payment required|insufficient[ _](?:quota|credits?|funds|balance)|credit balance|"
                       r"out of credits|spend(?:ing)? limit|(?:monthly|plan|account) quota|"
                       r"quota (?:exhausted|depleted)", re.I)
# How a kept-alive connection the server closed while idle fails when it is reused.
_STALE = (http.client.RemoteDisconnected, http.client.ImproperConnectionState, ConnectionResetError,
          ConnectionAbortedError, BrokenPipeError, ssl.SSLEOFError, ssl.SSLZeroReturnError)
_BACKOFF_S, _BACKOFF_CAP_S = 0.1, 2.0
_sleep = time.sleep          # module level, so tests can skip waits without patching the time module


def usage_guide() -> str:
    """Versioned model context for the proposer and runtime Astra prompt builders.

    Reads a bundled, credential-free snapshot; never fetches docs at inference time.
    """
    return Path(__file__).with_name("jev_usage.md").read_text(encoding="utf-8")


Instr = Any   # str | dict | list — structured instructions are allowed by the API


@dataclass
class Choice:
    instructions: Instr
    criteria: dict[str, Any]          # option -> description (str | structured | None)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice needs at least one option")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"Choice allows at most {MAX_CHOICE_OPTIONS} options")

    def to_json(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass
class Score:
    instructions: Instr
    criteria: list[Any]               # ordered levels, each a concrete situation

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("Score needs at least 2 levels")
        if len(self.criteria) > MAX_SCORE_LEVELS:   # the API rejects the request (422)
            raise ValueError(f"Score allows at most {MAX_SCORE_LEVELS} levels, got {len(self.criteria)}")

    def to_json(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass
class Noul:
    instructions: Instr
    criteria: dict[str, Any] = field(default_factory=dict)   # optional {"true": ..., "false": ...}

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


def _number(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{what} is not a finite number: {x!r}")
    return float(x)


def _probabilities(a: Mapping[str, Any]) -> dict[str, float]:
    p = a.get("probabilities")
    if p is None:
        return {}
    if not isinstance(p, Mapping):
        raise ValueError(f"probabilities is not an object: {p!r}")
    return {str(k): _number(v, f"probability of {k!r}") for k, v in p.items()}


def _confidence(a: Mapping[str, Any]) -> Optional[float]:
    c = a.get("confidence")
    return None if c is None else _number(c, "confidence")


def parse_answer(a: Mapping[str, Any]) -> System1Answer:
    """One API answer, checked: ValueError if it is malformed or of an unknown type."""
    if not isinstance(a, Mapping):
        raise ValueError(f"answer is not an object: {a!r}")
    t = a.get("type")
    if t == "choice":
        if not isinstance(a.get("choice"), str):
            raise ValueError(f"choice answer without a chosen option: {a!r}")
        return System1Answer("choice", a["choice"], _probabilities(a), _confidence(a))
    if t == "score":
        return System1Answer("score", _number(a.get("score"), "score"), _probabilities(a), _confidence(a))
    if t == "noul":
        p = _number(a.get("noul"), "noul")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"noul probability out of [0, 1]: {p}")
        return System1Answer("noul", p, {"true": p, "false": 1.0 - p}, _confidence(a))
    if t == "box":      # image questions (assets) — documented, though this client sends no images
        box = a.get("box")
        if not isinstance(box, Mapping):
            raise ValueError(f"box answer without a box: {a!r}")
        edges = {k: _number(box.get(k), f"box {k}") for k in ("top", "bottom", "left", "right")}
        return System1Answer("box", edges, {}, _confidence(a))
    raise ValueError(f"unknown answer type {t!r}")


class JevClient:
    """Jev over HTTP(S), standard library only.

    * One kept-alive connection per thread (one TCP + TLS handshake, not one per call). A kept-alive
      connection the server closed while idle fails on reuse and is replaced at once, without backoff.
    * ``timeout_s`` bounds the whole call — connect, send, every read and all backoff; running out
      raises ``TimeoutError`` naming the last error. (DNS lookups cannot be bounded in the stdlib.)
    * 408 / 429 / 5xx (529 included) and transport errors are retried with exponential backoff +
      jitter inside that deadline; ``Retry-After`` (seconds or HTTP date) is honoured when it fits.
    * 401 / 402 / 403, and a 4xx (429 included, 422 not) whose body names billing / quota ->
      ``ModelUnavailable`` (it stops a run). 400 / 413 / 415 / 422 and other 4xx -> ``RuntimeError``
      with the response detail, never retried.
    * Throttling becomes ``ModelUnavailable`` only when it persists: ``THROTTLED_CALLS_UNAVAILABLE``
      (5) consecutive calls of this client each ran out of time on 429/529 alone, spanning at least
      ``THROTTLED_SPAN_UNAVAILABLE_S`` (30 s), so a burst of parallel calls cannot stop a run. Any
      other outcome of a call, a success included, resets the count.
    * A body that is not JSON, or lacks a well-formed answer of the asked type for every question,
      is a traced ``RuntimeError`` — never a success with partial answers.

    HTTPS proxies from the environment / system settings are used through a CONNECT tunnel, as
    urllib did. ``model`` is required: model ids live in ``models.toml``.
    """
    name = "jev"

    def __init__(self, api_key: Optional[str] = None, *, model: str, url: str = JEV_URL) -> None:
        if not model:
            raise ValueError("JevClient needs a model id: set `model` in the typesafe profile of models.toml")
        load_dotenv()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise ModelUnavailable("TYPESAFE_API_KEY missing — put it in .env")
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"Jev URL must be http(s)://host/path, got {url!r}")
        self.model = model
        self.url = url
        self.total_input_tokens = 0
        self.n_calls = 0
        self._host = parts.hostname
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self._tls = ssl.create_default_context() if parts.scheme == "https" else None
        self._proxy = _https_proxy(self._host) if self._tls else None
        self._headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                         "Accept": "application/json"}
        self._local = threading.local()          # .conn: this thread's kept-alive connection
        self._lock = threading.Lock()
        self._conns: weakref.WeakSet[http.client.HTTPConnection] = weakref.WeakSet()
        self._throttled_calls = 0
        self._throttled_since = 0.0

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        timeout_s: float = 5.0,
        trace: Optional[TraceSink] = None,
    ) -> dict[str, System1Answer]:
        qjson = {k: _q_json(q) for k, q in questions.items()}
        body = {"model": self.model, "state": state, "questions": qjson}
        data = json.dumps(body).encode()
        t0, start = time.time(), time.monotonic()
        deadline = start + timeout_s
        attempts, last_err, throttled_only, saw_throttle = 0, None, True, False
        while time.monotonic() < deadline:
            attempts += 1
            try:
                status, headers, raw = self._post(data, deadline)
            except (OSError, http.client.HTTPException) as e:    # refused, reset, TLS, out of time, ...
                last_err = f"{type(e).__name__}: {e}"
                # the call's own deadline cutting a retry short is not a different failure
                if not (isinstance(e, TimeoutError) and time.monotonic() >= deadline - 0.01):
                    throttled_only = False
                self._pause(_backoff(attempts), deadline)
                continue
            if 200 <= status < 300:
                return self._answers(raw, qjson, body, t0, start, trace)
            detail = raw.decode("utf-8", "replace").strip()[:500]
            last_err = f"HTTP {status}: {detail}"
            # 422 details echo the request (game state may well say "billing"): never read as quota
            billing = 400 <= status < 500 and status != 422 and _QUOTA_RE.search(detail)
            if status in _UNAVAILABLE_STATUS or billing:
                self._outcome(False, start)
                self._log(trace, t0, body, None, last_err)
                raise ModelUnavailable(f"Jev {last_err}")
            if status not in _RETRY_STATUS:
                self._outcome(False, start)
                self._log(trace, t0, body, None, last_err)
                raise RuntimeError(f"Jev rejected the request: {last_err}")
            throttled_only = throttled_only and status in _THROTTLE_STATUS
            saw_throttle = saw_throttle or status in _THROTTLE_STATUS
            wait = _retry_after(headers.get("retry-after"))
            self._pause(wait if wait is not None and wait < deadline - time.monotonic() else _backoff(attempts),
                        deadline)
        err = (f"Jev request deadline ({timeout_s:g}s) exceeded after {attempts} attempt(s); "
               f"last error: {last_err or 'none'}")
        streak = self._outcome(saw_throttle and throttled_only, start)
        if streak:
            err = (f"Jev throttled (429/529) through {streak} consecutive calls over "
                   f"{THROTTLED_SPAN_UNAVAILABLE_S:g}s+; last error: {last_err}")
        self._log(trace, t0, body, None, err)
        raise ModelUnavailable(err) if streak else TimeoutError(err)

    def close(self) -> None:
        """Close every kept-alive connection of this client (all threads); the next call reconnects."""
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            conn.close()

    # -- transport ------------------------------------------------------
    def _post(self, data: bytes, deadline: float) -> tuple[int, dict[str, str], bytes]:
        """One POST on this thread's kept-alive connection (or a new one)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None and conn.sock is not None:
            try:
                return self._exchange(conn, data, deadline)
            except _STALE:
                self._drop(conn)        # closed by the server while idle: reconnect at once, no backoff
            except BaseException:
                self._drop(conn)
                raise
        conn = self._connect(deadline)
        try:
            return self._exchange(conn, data, deadline)
        except BaseException:
            self._drop(conn)
            raise

    def _new_connection(self, timeout: float) -> http.client.HTTPConnection:
        if self._tls is None:
            return http.client.HTTPConnection(self._host, self._port, timeout=timeout)
        if self._proxy is None:
            return http.client.HTTPSConnection(self._host, self._port, timeout=timeout, context=self._tls)
        proxy_host, proxy_port, proxy_headers = self._proxy
        conn = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=timeout, context=self._tls)
        conn.set_tunnel(self._host, self._port, headers=proxy_headers)
        return conn

    def _connect(self, deadline: float) -> http.client.HTTPConnection:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline reached before connecting")
        conn = self._new_connection(remaining)
        try:
            conn.connect()          # TCP (+ proxy CONNECT) + TLS handshake, bounded by the time left
        except BaseException:
            conn.close()
            raise
        self._local.conn = conn
        with self._lock:
            self._conns.add(conn)
        return conn

    def _exchange(self, conn: http.client.HTTPConnection, data: bytes,
                  deadline: float) -> tuple[int, dict[str, str], bytes]:
        """Send and read the whole response (a kept-alive connection must be drained), every
        socket operation limited to the time left of the call."""
        sock = conn.sock

        def left() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline reached")
            return remaining

        sock.settimeout(left())
        conn.request("POST", self._path, body=data, headers=self._headers)
        sock.settimeout(left())
        resp = conn.getresponse()
        chunks: list[bytes] = []
        while True:
            sock.settimeout(left())
            chunk = resp.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
        if resp.will_close:          # the server ends the connection after this response
            self._drop(conn)
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, b"".join(chunks)

    def _drop(self, conn: http.client.HTTPConnection) -> None:
        conn.close()
        if getattr(self._local, "conn", None) is conn:
            self._local.conn = None

    @staticmethod
    def _pause(seconds: float, deadline: float) -> None:
        _sleep(max(0.0, min(seconds, deadline - time.monotonic())))

    # -- outcomes -------------------------------------------------------
    def _outcome(self, throttled: bool, start: float) -> int:
        """Count consecutive calls that ran out of time on throttling alone (any other outcome
        resets); returns the count once it makes Jev unavailable, else 0."""
        with self._lock:
            if not throttled:
                self._throttled_calls = 0
                return 0
            if self._throttled_calls == 0:
                self._throttled_since = start
            self._throttled_calls += 1
            calls, span = self._throttled_calls, time.monotonic() - self._throttled_since
        return calls if calls >= THROTTLED_CALLS_UNAVAILABLE and span >= THROTTLED_SPAN_UNAVAILABLE_S else 0

    def _answers(self, raw: bytes, qjson: dict[str, dict[str, Any]], body: Any, t0: float, start: float,
                 trace: Optional[TraceSink]) -> dict[str, System1Answer]:
        payload: Any = None
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("the response is not a JSON object")
            answers = payload.get("answers")
            if not isinstance(answers, dict):
                raise ValueError("the response has no 'answers' object")
            missing = sorted(set(qjson) - set(answers))
            if missing:
                raise ValueError(f"no answer for {missing}")
            out: dict[str, System1Answer] = {}
            for qid, a in answers.items():
                ans = parse_answer(a)
                asked = qjson.get(qid, {}).get("type")
                if asked is not None and ans.type != asked:
                    raise ValueError(f"answer {qid!r} is a {ans.type}, the question a {asked}")
                out[qid] = ans
        except ValueError as e:        # json.JSONDecodeError / UnicodeDecodeError included
            self._outcome(False, start)
            err = f"malformed Jev response: {e}"
            self._log(trace, t0, body, payload if payload is not None else raw[:2000].decode("utf-8", "replace"), err)
            raise RuntimeError(err) from e
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tokens = usage.get("input_tokens")
        with self._lock:
            self._throttled_calls = 0
            self.n_calls += 1
            self.total_input_tokens += tokens if isinstance(tokens, int) else 0
        if trace:
            trace.log_call(ModelCall("system1", f"jev:{payload.get('model') or self.model}", t0, time.time() - t0,
                                     body, payload, usage=usage))
        return out

    def _log(self, trace: Optional[TraceSink], t0: float, body: Any, payload: Any, err: str | None) -> None:
        if trace:
            trace.log_call(ModelCall("system1", "jev", t0, time.time() - t0, body, payload, ok=err is None, error=err))


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: half of each step fixed, half random."""
    step = min(_BACKOFF_CAP_S, _BACKOFF_S * 2 ** (attempt - 1))
    return step / 2 + random.uniform(0, step / 2)


def _retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds a Retry-After header asks for (delta-seconds or HTTP date); None if absent or invalid."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _https_proxy(host: str) -> Optional[tuple[str, int, dict[str, str]]]:
    """The HTTP proxy urllib would tunnel ``host`` through (environment / system settings), if any."""
    proxy = urllib.request.getproxies().get("https")
    if not proxy or urllib.request.proxy_bypass(host):
        return None
    parts = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    if parts.scheme != "http" or not parts.hostname:
        return None
    headers: dict[str, str] = {}
    if parts.username:
        cred = f"{urllib.parse.unquote(parts.username)}:{urllib.parse.unquote(parts.password or '')}"
        headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode("ascii")
    return parts.hostname, parts.port or 8080, headers
