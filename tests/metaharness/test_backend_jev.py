"""Jev over a local HTTP/1.1 server: kept-alive connections, one deadline per call, retries,
Retry-After, throttling persistence and strict response checks (no network beyond 127.0.0.1)."""
from __future__ import annotations

import email.utils
import http.client
import json
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from metaharness.models import jev as jev_mod
from metaharness.models import make_system1
from metaharness.models.base import ModelUnavailable, TraceSink
from metaharness.models.jev import Choice, JevClient, Noul, Score

ANSWER = {"type": "choice", "choice": "up", "probabilities": {"up": 0.7, "down": 0.3}, "confidence": 0.5}
OK = {"model": "jev-1.13.0", "answers": {"move": ANSWER}, "usage": {"input_tokens": 10, "output_tokens": 1}}
Q = {"move": Choice("q", {"up": "u", "down": "d"})}


@dataclass
class Reply:
    status: int = 200
    body: Any = field(default_factory=lambda: OK)
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    trickle: float = 0.0        # seconds between body bytes (0: the body in one write)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 5.0       # idle kept-alive connections end on their own

    def setup(self) -> None:
        super().setup()
        with self.server.lock:
            self.server.connections += 1

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        srv = self.server
        with srv.lock:
            srv.requests.append({"body": json.loads(raw), "auth": self.headers.get("Authorization"),
                                 "path": self.path, "port": self.client_address[1]})
            reply = srv.script.pop(0) if srv.script else srv.default
        if reply.delay:
            time.sleep(reply.delay)
        data = reply.body if isinstance(reply.body, bytes) else json.dumps(reply.body).encode()
        self.send_response(reply.status)
        for k, v in reply.headers.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not reply.trickle:
            self.wfile.write(data)
            return
        for i in range(len(data)):      # until done, or the client hangs up
            self.wfile.write(data[i:i + 1])
            time.sleep(reply.trickle)

    def log_message(self, *args: Any) -> None:
        pass


class JevServer(ThreadingHTTPServer):
    """Scripted stand-in for the System-One endpoint: ``script`` replies in order, then ``default``."""
    daemon_threads = True
    block_on_close = False

    def __init__(self, handler_timeout: float = 5.0) -> None:
        super().__init__(("127.0.0.1", 0), type("Handler", (_Handler,), {"timeout": handler_timeout}))
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.script: list[Reply] = []
        self.default = Reply()
        self.connections = 0
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1/systemone"

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass      # a client that gave up mid-reply is expected here

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.fixture
def jev_server():
    srv = JevServer()
    yield srv
    srv.stop()


@pytest.fixture
def make_client(jev_server):
    clients: list[JevClient] = []

    def make(server: JevServer | None = None) -> JevClient:
        c = JevClient(api_key="k", model="jev-test", url=(server or jev_server).url)
        clients.append(c)
        return c
    yield make
    for c in clients:
        c.close()


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff waits instead of sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(jev_mod, "_sleep", waits.append)
    return waits


def test_jev_reuses_one_connection_per_thread(jev_server, make_client):
    c = make_client()
    tr = TraceSink()
    for i in range(3):
        assert c.decide({"i": i}, Q, trace=tr)["move"].value == "up"
    assert jev_server.connections == 1 and len({r["port"] for r in jev_server.requests}) == 1
    req = jev_server.requests[0]
    assert req["auth"] == "Bearer k" and req["path"] == "/v1/systemone"
    assert req["body"] == {"model": "jev-test", "state": {"i": 0}, "questions": {"move": Q["move"].to_json()}}
    assert c.n_calls == 3 and c.total_input_tokens == 30
    assert [r["ok"] for r in tr.records] == [True] * 3 and tr.records[0]["backend"] == "jev:jev-1.13.0"
    worker = threading.Thread(target=lambda: c.decide({}, Q))     # another thread: its own connection
    worker.start()
    worker.join()
    assert jev_server.connections == 2


def test_jev_parallel_threads_keep_their_connections(jev_server, make_client):
    c = make_client()
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(5):
                c.decide({}, Q)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    workers = [threading.Thread(target=work) for _ in range(6)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors and c.n_calls == 30 and c.total_input_tokens == 300
    assert jev_server.connections == 6


def test_jev_reconnects_at_once_after_server_closed_idle_connection(make_client, sleeps):
    srv = JevServer(handler_timeout=0.2)     # the server drops a connection idle for 0.2 s
    try:
        c = make_client(srv)
        assert c.decide({}, Q)["move"].value == "up"
        time.sleep(0.6)
        assert c.decide({}, Q)["move"].value == "up"
        assert srv.connections == 2 and len(srv.requests) == 2
        assert sleeps == []                  # the stale connection was replaced without backoff
    finally:
        srv.stop()


def test_jev_honours_retry_after_then_succeeds(jev_server, make_client, sleeps):
    jev_server.script = [Reply(429, {"detail": "slow down"}, {"Retry-After": "0.3"})]
    tr = TraceSink()
    assert make_client().decide({}, Q, trace=tr)["move"].value == "up"
    assert len(jev_server.requests) == 2 and sleeps == [pytest.approx(0.3)]
    assert [r["ok"] for r in tr.records] == [True]


def test_jev_retry_after_beyond_the_deadline_falls_back_to_backoff(jev_server, make_client, sleeps):
    jev_server.script = [Reply(429, {"detail": "slow down"}, {"Retry-After": "120"})]
    assert make_client().decide({}, Q, timeout_s=5)["move"].value == "up"
    assert len(sleeps) == 1 and sleeps[0] < 1.0


def test_retry_after_header_forms():
    when = datetime.now(timezone.utc) + timedelta(seconds=10)
    assert 8 <= jev_mod._retry_after(email.utils.format_datetime(when, usegmt=True)) <= 10
    assert jev_mod._retry_after("2.5") == 2.5 and jev_mod._retry_after("-3") == 0.0
    assert jev_mod._retry_after(None) is None and jev_mod._retry_after("soon") is None
    assert jev_mod._retry_after("nan") is None


def test_jev_retries_5xx_and_408(jev_server, make_client, sleeps):
    jev_server.script = [Reply(503, {"detail": "unavailable"}), Reply(408, b""), Reply(529, {"detail": "overloaded"})]
    assert make_client().decide({}, Q)["move"].value == "up"
    assert len(jev_server.requests) == 4 and len(sleeps) == 3


def test_jev_whole_call_obeys_one_deadline(jev_server, make_client):
    jev_server.default = Reply(delay=2.0)          # a reply that comes too late
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match=r"deadline \(0.4s\) exceeded.*TimeoutError"):
        make_client().decide({}, Q, timeout_s=0.4)
    assert time.monotonic() - t0 < 1.5


def test_jev_deadline_holds_against_a_trickling_response(jev_server, make_client):
    """A socket timeout bounds each recv: a body arriving a byte every 50 ms (~10 s in all) kept a 1 s
    call alive until its end. The connection cut at the deadline is dropped; the next call reconnects."""
    jev_server.default = Reply(trickle=0.05)
    c = make_client()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match=r"deadline \(1s\) exceeded.*cut off"):
        c.decide({}, Q, timeout_s=1)
    assert time.monotonic() - t0 < 1.5
    assert getattr(c._local, "conn", None) is None
    jev_server.default = Reply()
    assert c.decide({}, Q)["move"].value == "up" and jev_server.connections == 2


def test_jev_deadline_holds_against_a_trickling_proxy(monkeypatch):
    """Connecting is cut off too: here a proxy answers CONNECT a byte every 50 ms (~2 s in all)."""
    listener = socket.create_server(("127.0.0.1", 0))

    def proxy() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            for byte in b"HTTP/1.1 200 Connection established\r\n\r\n":
                try:
                    conn.sendall(bytes([byte]))
                except OSError:
                    return
                time.sleep(0.05)
    threading.Thread(target=proxy, daemon=True).start()
    monkeypatch.setattr(jev_mod.urllib.request, "getproxies",
                        lambda: {"https": f"http://127.0.0.1:{listener.getsockname()[1]}"})
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: False)
    c = JevClient(api_key="k", model="m", url="https://jev.invalid/v1/systemone")    # never resolved here
    t0 = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match=r"deadline \(1s\) exceeded.*cut off"):
            c.decide({}, Q, timeout_s=1)
        assert time.monotonic() - t0 < 1.5 and getattr(c._local, "conn", None) is None
    finally:
        c.close()
        listener.close()


def test_jev_deadline_names_the_last_error(jev_server, make_client):
    jev_server.default = Reply(503, {"detail": "maintenance"})
    tr = TraceSink()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="HTTP 503.*maintenance"):
        make_client().decide({}, Q, timeout_s=0.4, trace=tr)
    assert time.monotonic() - t0 < 1.5 and len(jev_server.requests) >= 2
    assert tr.records[-1]["ok"] is False and "deadline" in tr.records[-1]["error"]


def test_jev_throttling_is_unavailable_only_when_it_persists(jev_server, make_client, monkeypatch):
    jev_server.default = Reply(429, {"detail": "rate limited"})
    c = make_client()
    for _ in range(jev_mod.THROTTLED_CALLS_UNAVAILABLE + 1):     # a quick burst is never fatal
        with pytest.raises(TimeoutError):
            c.decide({}, Q, timeout_s=0.12)
    monkeypatch.setattr(jev_mod, "THROTTLED_SPAN_UNAVAILABLE_S", 0.0)
    jev_server.default = Reply()
    c.decide({}, Q)                                   # a success resets the count
    jev_server.default = Reply(529, {"detail": "overloaded"})
    for _ in range(jev_mod.THROTTLED_CALLS_UNAVAILABLE - 1):
        with pytest.raises(TimeoutError):
            c.decide({}, Q, timeout_s=0.12)
    tr = TraceSink()
    with pytest.raises(ModelUnavailable, match="throttled"):
        c.decide({}, Q, timeout_s=0.12, trace=tr)
    assert tr.records[-1]["ok"] is False


def test_jev_mixed_failures_do_not_count_as_throttling(jev_server, make_client, monkeypatch):
    monkeypatch.setattr(jev_mod, "THROTTLED_SPAN_UNAVAILABLE_S", 0.0)
    c = make_client()
    for _ in range(jev_mod.THROTTLED_CALLS_UNAVAILABLE + 1):
        jev_server.script = [Reply(429, {"detail": "rate limited"}), Reply(503, {"detail": "down"})]
        jev_server.default = Reply(503, {"detail": "down"})
        with pytest.raises(TimeoutError):
            c.decide({}, Q, timeout_s=0.12)


@pytest.mark.parametrize("status,body", [
    (401, {"detail": "Invalid API key"}),
    (403, {"detail": "Forbidden"}),
    (402, {"detail": "Payment required"}),
    (429, {"detail": "Monthly quota exhausted; update your billing details"}),
    (400, {"detail": "Insufficient credits for this request"}),
])
def test_jev_auth_and_billing_are_unavailable_at_once(jev_server, make_client, status, body):
    jev_server.default = Reply(status, body)
    tr = TraceSink()
    with pytest.raises(ModelUnavailable, match=str(status)):
        make_client().decide({}, Q, trace=tr)
    assert len(jev_server.requests) == 1 and tr.records[-1]["ok"] is False


@pytest.mark.parametrize("status", [400, 404, 413, 415, 422])
def test_jev_rejected_requests_are_not_retried(jev_server, make_client, status):
    detail = {"detail": [{"loc": ["body", "questions", "move", "criteria"], "msg": "too many items"}]}
    if status == 422:     # a validation error echoes the input: its words are not a billing verdict
        detail["detail"][0]["input"] = {"state": "the shop's billing desk is out of credits"}
    jev_server.default = Reply(status, detail)
    tr = TraceSink()
    with pytest.raises(RuntimeError, match=f"HTTP {status}.*criteria") as ei:
        make_client().decide({}, Q, trace=tr)
    assert not isinstance(ei.value, ModelUnavailable)
    assert len(jev_server.requests) == 1 and tr.records[-1]["ok"] is False


@pytest.mark.parametrize("body", [
    b"<html>gateway</html>",
    [1, 2],
    {"model": "jev-1.13.0"},
    {"answers": {"other": ANSWER}},
    {"answers": {"move": {"type": "mystery", "value": 1}}},
    {"answers": {"move": {"type": "noul", "noul": 0.4}}},
    {"answers": {"move": {"type": "choice", "probabilities": {"up": 1.0}}}},
    {"answers": {"move": {**ANSWER, "probabilities": {"up": "high"}}}},
])
def test_jev_malformed_responses_are_traced_failures(jev_server, make_client, body):
    jev_server.default = Reply(200, body)
    c = make_client()
    tr = TraceSink()
    with pytest.raises(RuntimeError, match="malformed Jev response"):
        c.decide({}, Q, trace=tr)
    assert tr.records[-1]["ok"] is False and c.n_calls == 0 and len(jev_server.requests) == 1


def test_jev_answers_every_question_type(jev_server, make_client):
    jev_server.default = Reply(200, {"answers": {
        "move": ANSWER,
        "risk": {"type": "score", "score": 1, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}, "confidence": 0.4},
        "safe": {"type": "noul", "noul": 0.9}}})          # no model / usage fields: tolerated
    c = make_client()
    ans = c.decide({}, {**Q, "risk": Score("q", ["lo", "mid", "hi"]), "safe": Noul("q")})
    assert ans["move"].value == "up" and ans["risk"].value == 1.0 and ans["safe"].probabilities["false"] == pytest.approx(0.1)
    assert c.n_calls == 1 and c.total_input_tokens == 0


def test_score_level_limits():
    Score("q", [str(i) for i in range(10)])
    with pytest.raises(ValueError, match="at most 10"):
        Score("q", [str(i) for i in range(11)])
    with pytest.raises(ValueError):
        Score("q", ["only"])


def test_jev_model_comes_from_the_profile(tmp_path, monkeypatch):
    with pytest.raises(TypeError):
        JevClient(api_key="k")                       # no default model id in code
    with pytest.raises(ValueError, match="model"):
        JevClient(api_key="k", model="")
    profiles = tmp_path / "models.toml"
    profiles.write_text('[system1.nomodel]\nbackend = "typesafe"\n\n[system1.jev2]\nbackend = "typesafe"\nmodel = "jev-x"\n')
    monkeypatch.setenv("METAHARNESS_MODELS", str(profiles))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    with pytest.raises(ValueError, match="nomodel.*needs a model id"):
        make_system1("nomodel")
    assert make_system1("jev2").model == "jev-x"


def test_jev_uses_the_https_proxy_urllib_would(monkeypatch):
    monkeypatch.setattr(jev_mod.urllib.request, "getproxies", lambda: {"https": "http://u:p%40ss@proxy.local:3128"})
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: False)
    conn = JevClient(api_key="k", model="m")._new_connection(1.0)      # not connected
    assert (conn.host, conn.port) == ("proxy.local", 3128)
    assert (conn._tunnel_host, conn._tunnel_port) == ("api.typesafe.ai", 443)
    assert conn._tunnel_headers["Proxy-Authorization"] == "Basic dTpwQHNz"
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: True)
    direct = JevClient(api_key="k", model="m")._new_connection(1.0)
    assert direct.host == "api.typesafe.ai" and direct._tunnel_host is None


def _urllib_route(monkeypatch, proxy: str) -> tuple[str, int, str, int]:
    """Where urllib itself sends an HTTPS request through ``proxy`` (its connect stubbed: no network)."""
    seen: list[tuple[str, int, str, int]] = []

    def connect(conn: http.client.HTTPSConnection) -> None:
        seen.append((conn.host, conn.port, conn._tunnel_host, conn._tunnel_port))
        raise OSError("not connecting")
    with monkeypatch.context() as m:
        m.setattr(http.client.HTTPSConnection, "connect", connect)
        with pytest.raises(OSError):
            urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy})).open(jev_mod.JEV_URL, timeout=1)
    return seen[-1]


@pytest.mark.parametrize("proxy,host,port", [
    ("http://proxy.local:3128", "proxy.local", 3128),
    ("https://proxy.local:3129", "proxy.local", 3129),     # urllib sent a plaintext CONNECT to it all the same
    ("http://proxy.local", "proxy.local", 443),            # no port: 443, urllib's HTTPSConnection default
    ("https://proxy.local", "proxy.local", 443),
    ("proxy.local:3130", "proxy.local", 3130),
    ("http://[::1]:3131", "::1", 3131),
])
def test_jev_tunnels_where_urllib_did(monkeypatch, proxy, host, port):
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(jev_mod.urllib.request, "getproxies", lambda: {"https": proxy})
    conn = JevClient(api_key="k", model="m")._new_connection(1.0)      # not connected
    route = (conn.host, conn.port, conn._tunnel_host, conn._tunnel_port)
    assert isinstance(conn, http.client.HTTPSConnection)
    assert route == (host, port, "api.typesafe.ai", 443) == _urllib_route(monkeypatch, proxy)


@pytest.mark.parametrize("proxy", ["socks5://proxy.local:1080", "socks5h://user:s3cret@proxy.local",
                                   "ftp://proxy.local"])
def test_jev_refuses_a_proxy_it_cannot_tunnel_through(monkeypatch, proxy):
    monkeypatch.setattr(jev_mod.urllib.request, "getproxies", lambda: {"https": proxy})
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: False)
    with pytest.raises(ValueError, match="unsupported https proxy .*proxy.local") as ei:
        JevClient(api_key="k", model="m")              # never a silent direct connection
    assert "s3cret" not in str(ei.value)
    monkeypatch.setattr(jev_mod.urllib.request, "proxy_bypass", lambda host: True)
    assert JevClient(api_key="k", model="m")._proxy is None     # bypassed for this host: not used, no error


def test_jev_deadline_holds_against_a_trickling_close_delimited_response(jev_server, make_client):
    # Connection: close hands the socket to the response: conn.sock is None while the body is read
    jev_server.default = Reply(trickle=0.05, headers={"Connection": "close"})
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        make_client().decide({}, Q, timeout_s=1)
    assert time.monotonic() - t0 < 1.5
