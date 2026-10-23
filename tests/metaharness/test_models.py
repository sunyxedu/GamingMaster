"""Offline tests for the model adapters (request shapes, response parsing,
failure classification). Live Jev test runs only when TYPESAFE_API_KEY is set."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from metaharness.models import codex_cli as codex_mod
from metaharness.models import make_system1, make_system2
from metaharness.models.codex_cli import CodexCLI, _digest_events, _last_agent_message, _parse_jsonl
from metaharness.models.base import ModelUnavailable, TraceSink
from metaharness.models.jev import Choice, JevClient, Noul, Score, parse_answer

from .test_backend_jev import JevServer, Reply


# ---------------------------------------------------------------- Jev

def test_question_json_shapes():
    assert Choice("q", {"a": "A", "b": "B"}).to_json() == {"type": "choice", "instructions": "q", "criteria": {"a": "A", "b": "B"}}
    assert Score("q", ["lo", "hi"]).to_json() == {"type": "score", "instructions": "q", "criteria": ["lo", "hi"]}
    assert Noul("q").to_json() == {"type": "noul", "instructions": "q"}


def test_parse_answers():
    a = parse_answer({"type": "choice", "choice": "up", "probabilities": {"up": 0.75, "down": 0.25}, "confidence": 0.6})
    assert a.value == "up" and a.top(1) == [("up", 0.75)]
    s = parse_answer({"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}})
    assert s.value == 1.4
    n = parse_answer({"type": "noul", "noul": 0.9})
    assert n.value == 0.9 and abs(n.probabilities["false"] - 0.1) < 1e-9


def test_jev_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("metaharness.models.jev.load_dotenv", lambda *a, **k: {})
    with pytest.raises(ModelUnavailable):
        JevClient(model="jev-test")


def test_jev_request_body_and_retry():
    """Local HTTP/1.1 server: the first call gets a 429, the retry a well-formed answer."""
    srv = JevServer()
    srv.script = [Reply(429, {"detail": "slow down"}, {"Retry-After": "0"})]
    c = JevClient(api_key="k", model="jev-latest", url=srv.url)
    try:
        tr = TraceSink()
        ans = c.decide({"x": 1}, {"move": Choice("q", {"up": "u", "down": "d"})}, trace=tr)
    finally:
        c.close()
        srv.stop()
    assert ans["move"].value == "up"
    calls = [r["body"] for r in srv.requests]
    assert len(calls) == 2 and calls[0]["model"] == "jev-latest" and calls[0]["questions"]["move"]["type"] == "choice"
    assert all(r["auth"] == "Bearer k" for r in srv.requests) and srv.connections == 1
    assert c.total_input_tokens == 10
    assert tr.records[-1]["kind"] == "system1" and tr.records[-1]["backend"] == "jev:jev-1.13.0"


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY") and not (Path(__file__).resolve().parents[2] / ".env").exists(),
                    reason="no Jev key")
def test_jev_live_snake_decision():
    try:
        c = make_system1("jev")     # the model id comes from models.toml
    except ModelUnavailable as e:   # a .env made for other settings (ARC) has no key
        pytest.skip(f"no Jev key: {e}")
    ans = c.decide(
        {"head": [5, 5], "food_delta": [0, -3], "danger": {"up": False, "down": False, "left": True, "right": False}},
        {"move": Choice("Which safe heading moves toward the food?", {"up": "toward food, safe", "down": "away, safe",
                                                                       "right": "sideways, safe"})},
        timeout_s=15,
    )
    assert ans["move"].value in ("up", "down", "right")
    assert abs(sum(ans["move"].probabilities.values()) - 1.0) < 0.05


# ---------------------------------------------------------------- profiles


def test_profiles_resolve_backend_and_model():
    m = make_system2("astra")
    assert isinstance(m, CodexCLI) and m.model == "gpt-6-astra" and m.reasoning_effort == "high"
    assert make_system2("astra", reasoning_effort="low").reasoning_effort == "low"
    assert make_system2("sol").model == "gpt-5.6-sol"
    from metaharness.models.claude_cli import ClaudeCLI
    c = make_system2("claude")
    assert isinstance(c, ClaudeCLI) and c.model is None
    with pytest.raises(ValueError):
        make_system2("no-such-profile")


# ---------------------------------------------------------------- codex CLI backend

def test_astra_argv_new_and_resume(tmp_path):
    a = CodexCLI(model="gpt-6-astra", reasoning_effort="high")
    argv = a._argv(cwd=tmp_path, session_id=None, agentic=False, last_msg_path=tmp_path / "l")
    assert argv[:2] == ["codex", "exec"] and "-m" in argv and argv[argv.index("-m") + 1] == "gpt-6-astra"
    assert "-s" in argv and "read-only" in argv and "--json" in argv
    assert argv[-1] == "-" and argv.count("-") == 1       # the prompt comes from stdin, never argv
    assert any(x == 'model_reasoning_effort="high"' for x in argv)
    argv2 = a._argv(cwd=tmp_path, session_id="abc", agentic=True, last_msg_path=tmp_path / "l")
    assert argv2[:4] == ["codex", "exec", "resume", "abc"] and argv2[argv2.index("-m") + 1] == "gpt-6-astra"
    assert argv2[-1] == "-"                               # `codex exec resume <id> -`: prompt from stdin
    assert "--color" not in argv2 and "-s" not in argv2 and "-C" not in argv2   # resume rejects these (live 0.154)
    argv3 = a._argv(cwd=tmp_path, session_id=None, agentic=True, last_msg_path=tmp_path / "l")
    assert "workspace-write" in argv3
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv3


CODEX_LIMIT_STDOUT = """{"type":"thread.started","thread_id":"01a0b6d5-ca9d-7c32-a13d-9189e0fc5393"}
{"type":"turn.started"}
{"type":"error","message":"You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 10:56 AM."}
{"type":"turn.failed","error":{"message":"You've hit your usage limit."}}
"""

CODEX_OK_STDOUT = """{"type":"thread.started","thread_id":"t-1"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"i1","type":"reasoning","text":"thinking"}}
{"type":"item.completed","item":{"id":"i2","type":"agent_message","text":"up"}}
{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}
"""


def test_codex_event_digest():
    ev = _parse_jsonl(CODEX_OK_STDOUT)
    tid, usage, errs = _digest_events(ev)
    assert tid == "t-1" and usage["input_tokens"] == 12 and errs == []
    assert _last_agent_message(ev) == "up"
    tid, usage, errs = _digest_events(_parse_jsonl(CODEX_LIMIT_STDOUT))
    assert tid and len(errs) == 2


def _fake_run_cli(stdout, rc, last_text=None):
    import subprocess

    def run_cli(argv, *, cwd, stdin_text, timeout_s, env_overrides=None, env_drop=()):
        assert "OPENAI_API_KEY" in env_drop
        assert cwd is None or Path(cwd).is_absolute()
        assert "-C" not in argv or Path(argv[argv.index("-C") + 1]).is_absolute()
        if last_text is not None:
            Path(argv[argv.index("-o") + 1]).write_text(last_text)
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")
    return run_cli


def test_astra_usage_limit_raises_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(CODEX_LIMIT_STDOUT, 1))
    tr = TraceSink()
    with pytest.raises(ModelUnavailable):
        CodexCLI(model="gpt-6-astra").run("hi", cwd=tmp_path, trace=tr)
    assert tr.records[-1]["ok"] is False


def test_astra_ok_returns_text_and_session(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(CODEX_OK_STDOUT, 0, last_text="up\n"))
    res = CodexCLI(model="gpt-6-astra").run("hi", cwd=tmp_path)
    assert res.text.strip() == "up" and res.session_id == "t-1" and res.usage["input_tokens"] == 12


@pytest.mark.parametrize('stdout,rc,last_text', [
    (CODEX_OK_STDOUT, 1, 'partial'),
    ('{"type":"turn.failed","error":{"message":"backend down"}}', 0, 'partial'),
    ('{"type":"turn.completed","usage":{}}', 0, ''),
])
def test_codex_incomplete_response_is_not_success(monkeypatch, stdout, rc, last_text):
    monkeypatch.setattr(codex_mod, 'run_cli', _fake_run_cli(stdout, rc, last_text))
    with pytest.raises(RuntimeError):
        CodexCLI(model='gpt-6-astra').run('test')


def test_cli_stdin_and_timeout(tmp_path):
    import sys
    from metaharness.models._cli import run_cli
    r = run_cli([sys.executable, '-c', 'import sys; print(sys.stdin.read().upper())'],
                cwd=tmp_path, stdin_text='hello', timeout_s=2)
    assert r.returncode == 0 and r.stdout.strip() == 'HELLO'
    with pytest.raises(TimeoutError, match='CLI invocation exceeded'):
        run_cli([sys.executable, '-c', 'import time; time.sleep(10)'],
                cwd=tmp_path, stdin_text=None, timeout_s=.05)


def test_player_external_tools_invalidate_inference(monkeypatch):
    tool_event = '{"type":"item.completed","item":{"type":"command_execution","command":"cat simulator.py"}}\n'
    monkeypatch.setattr(codex_mod, 'run_cli', _fake_run_cli(tool_event + CODEX_OK_STDOUT, 0, 'up'))
    model = CodexCLI(model='gpt-6-astra')
    with pytest.raises(ModelUnavailable, match='external tools'):
        model.run('play the game')
    # The proposer deliberately has tools; that is the meta-harness outer loop.
    assert model.run('design a harness', agentic=True).text == 'up'


def test_astra_argv_never_inherits_fast_tier(tmp_path):
    from metaharness.models.codex_cli import CodexCLI
    a = CodexCLI(model="m")
    for agentic in (False, True):
        argv = a._argv(cwd=tmp_path, session_id=None, agentic=agentic, last_msg_path=tmp_path / "l")
        assert 'service_tier="default"' in argv and "features.fast_mode=false" in argv
    player = a._argv(cwd=tmp_path, session_id=None, agentic=False, last_msg_path=tmp_path / "l")
    assert "project_doc_max_bytes=0" in player
