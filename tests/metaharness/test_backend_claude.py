"""claude backend through a fake ``claude`` CLI: player isolation, agentic restrictions, --effort,
quota/auth classification, traced failures, resumable timeouts, images via stream-json."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from metaharness.models import make_system2
from metaharness.models._cli import cancel_scope
from metaharness.models.base import (
    PLAYER_PREAMBLE,
    ModelCallCancelled,
    ModelTimeout,
    ModelUnavailable,
    TraceSink,
)
from metaharness.models.claude_cli import EFFORT_LEVELS, ClaudeCLI

WATCHED_ENV = ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "TYPESAFE_API_KEY",
               "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

# Logs argv / cwd / stdin / env, then answers per FAKE_CLAUDE_MODE.
FAKE_CLAUDE = """
import json, os, sys, time
argv = sys.argv[1:]
stdin = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "stdin": stdin,
                         "env": {k: os.environ.get(k) for k in %(watched)r}}) + "\\n")
mode, text = os.environ.get("FAKE_CLAUDE_MODE", "ok"), os.environ.get("FAKE_CLAUDE_TEXT", "up")
rc = int(os.environ.get("FAKE_CLAUDE_RC", "1"))
sid = (argv[argv.index("--resume") + 1] if "--resume" in argv
       else argv[argv.index("--session-id") + 1] if "--session-id" in argv else "generated")
result = {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": sid,
          "total_cost_usd": 0.01, "usage": {"input_tokens": 5, "output_tokens": 1}}
if mode == "ok":
    print(json.dumps(result))
elif mode == "ok_then_fail":
    print(json.dumps(result))
    sys.exit(rc)
elif mode == "verbose_array":
    print(json.dumps([{"type": "system", "subtype": "init", "session_id": sid}, result]))
elif mode == "stream":
    content = json.loads(stdin.splitlines()[0])["message"]["content"]
    seen = [b["source"]["media_type"] for b in content if b["type"] == "image"]
    print(json.dumps({"type": "system", "subtype": "init", "session_id": sid}))
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}))
    print(json.dumps({**result, "result": " ".join(seen) + " | " + content[-1]["text"]}))
elif mode == "error":
    print(json.dumps({**result, "is_error": True, "result": text}))
    sys.exit(rc)
elif mode == "errors":
    print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "errors": [text]}))
    sys.exit(rc)
elif mode == "stderr":
    print(text, file=sys.stderr)
    sys.exit(rc)
elif mode == "nonjson":
    print("not json at all")
elif mode == "empty":
    print(json.dumps({**result, "result": ""}))
elif mode == "sleep":
    time.sleep(30)
""" % {"watched": WATCHED_ENV}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE}")
    script.chmod(0o755)
    log = tmp_path / "claude-calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    def mode(name: str, text: str | None = None, rc: int = 1) -> None:
        monkeypatch.setenv("FAKE_CLAUDE_MODE", name)
        monkeypatch.setenv("FAKE_CLAUDE_RC", str(rc))
        if text is not None:
            monkeypatch.setenv("FAKE_CLAUDE_TEXT", text)
    return SimpleNamespace(bin=str(script), mode=mode, calls=lambda: [
        json.loads(line) for line in log.read_text().splitlines()] if log.exists() else [])


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_player_is_isolated_from_the_users_claude_setup(fake, monkeypatch):
    for key in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "CLAUDECODE"):
        monkeypatch.setenv(key, "leak")
    model, tr = ClaudeCLI(model="some-model", claude_bin=fake.bin), TraceSink()
    first = model.run("what next?", system="rules", trace=tr)
    second = model.run("and now?", session_id=first.session_id, trace=tr)
    c1, c2 = fake.calls()
    argv = c1["argv"]
    assert argv[:3] == ["-p", "--output-format", "json"] and _flag(argv, "--model") == "some-model"
    assert _flag(argv, "--tools") == "" and "--strict-mcp-config" in argv and "--disable-slash-commands" in argv
    assert json.loads(_flag(argv, "--settings")) == {"disableAllHooks": True}
    assert _flag(argv, "--system-prompt") == f"{PLAYER_PREAMBLE}\n\nrules"
    assert "--permission-mode" not in argv and "--bare" not in argv
    assert first.session_id == str(uuid.UUID(_flag(argv, "--session-id")))
    assert c1["stdin"] == "what next?"
    assert c1["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and c1["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert all(c1["env"][k] is None for k in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "CLAUDECODE"))
    player_dir = Path(c1["cwd"])
    assert player_dir.name.startswith("metaharness-claude-player-") and not any(player_dir.iterdir())
    assert c2["cwd"] == c1["cwd"]                          # sessions are stored per directory
    assert _flag(c2["argv"], "--resume") == first.session_id and "--session-id" not in c2["argv"]
    assert _flag(c2["argv"], "--system-prompt").startswith(PLAYER_PREAMBLE)
    assert second.text == "up" and model.player_tools == "disabled" and model.n_calls == 2
    assert tr.records[0]["backend"] == model.name and tr.records[0]["request"]["system"].startswith(PLAYER_PREAMBLE)
    del model
    gc.collect()
    assert not player_dir.exists()                         # removed with its client


def test_agentic_calls_keep_tools_but_not_web_unless_asked(fake, tmp_path):
    model = ClaudeCLI(claude_bin=fake.bin, reasoning_effort="xhigh")
    model.run("design", cwd=tmp_path, agentic=True, system="SYS")
    model.run("research", cwd=tmp_path, agentic=True, web_search=True)
    model.run("again", cwd=tmp_path, agentic=True, session_id="abc", system="SYS")
    c1, c2, c3 = fake.calls()
    assert _flag(c1["argv"], "--permission-mode") == "bypassPermissions" and "--tools" not in c1["argv"]
    assert c1["argv"][c1["argv"].index("--disallowedTools") + 1:][:2] == ["WebSearch", "WebFetch"]
    assert "--disallowedTools" not in c2["argv"]
    assert _flag(c1["argv"], "--system-prompt") == "SYS" and "--system-prompt" not in c3["argv"]
    assert c1["cwd"] == str(tmp_path.resolve()) and c1["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] is None
    assert all(_flag(c["argv"], "--effort") == "xhigh" for c in (c1, c2, c3))
    assert _flag(c3["argv"], "--resume") == "abc"


def test_effort_is_validated_at_construction():
    for level in EFFORT_LEVELS:
        assert ClaudeCLI(reasoning_effort=level).reasoning_effort == level
    with pytest.raises(ValueError, match="low, medium, high, xhigh, max"):
        ClaudeCLI(reasoning_effort="minimal")
    with pytest.raises(ValueError):
        make_system2("claude", reasoning_effort="ultra")
    assert ClaudeCLI(reasoning_effort="").reasoning_effort is None


UNAVAILABLE = [
    "Claude AI usage limit reached|1761140000",
    "You've hit your limit · resets 5pm (Europe/Berlin)",
    "You've reached your Fable limit",
    "Your session limit resets at 3pm",
    "You have used 100% of your weekly limit",
    "You're out of extra usage",
    "Your seat type doesn't include extra usage",
    "Fable 5 requires usage credits",
    "Credit balance is too low",
    "Not logged in · Please run /login",
    "Invalid API key · Fix external API key",
    "Invalid auth token · Fix external auth token",
    "OAuth token has expired. Please obtain a new token or refresh your existing token.",
    "Authentication required · Sign in again to continue",
]
TRANSIENT = [
    'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
    "API Error: 500 Internal server error",
    "API Error: Request timed out.",
]


@pytest.mark.parametrize("mode", ["error", "errors", "stderr"])
@pytest.mark.parametrize("message", UNAVAILABLE)
def test_quota_and_auth_failures_are_unavailable(fake, mode, message):
    fake.mode(mode, message)
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    with pytest.raises(ModelUnavailable, match="matched"):
        model.run("hi", trace=tr)
    assert tr.records[-1]["ok"] is False and tr.records[-1]["backend"] == model.name
    assert message[:20] in tr.records[-1]["error"]


@pytest.mark.parametrize("message", TRANSIENT)
def test_overload_and_server_errors_stay_runtime_errors(fake, message):
    fake.mode("error", message)
    with pytest.raises(RuntimeError) as ei:
        ClaudeCLI(claude_bin=fake.bin).run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


def test_is_error_with_exit_code_zero_is_a_failure(fake):
    fake.mode("error", "Not logged in · Please run /login", rc=0)
    with pytest.raises(ModelUnavailable):
        ClaudeCLI(claude_bin=fake.bin).run("hi")


def test_successful_answer_mentioning_limits_is_a_success(fake):
    answer = "You've hit your usage limit? No - the snake hit the wall. Not logged in to anything. up"
    fake.mode("ok", answer)
    assert ClaudeCLI(claude_bin=fake.bin).run("hi").text.endswith("up")
    fake.mode("ok_then_fail", answer)      # a failed exit after an answer: the answer is not the error
    with pytest.raises(RuntimeError, match="exit status 1") as ei:
        ClaudeCLI(claude_bin=fake.bin).run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


@pytest.mark.parametrize("mode,match", [("nonjson", "no JSON result"), ("empty", "empty result")])
def test_unusable_output_is_a_traced_runtime_error(fake, mode, match):
    fake.mode(mode)
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    with pytest.raises(RuntimeError, match=match):
        model.run("hi", trace=tr)
    assert tr.records[-1]["ok"] is False and model.n_calls == 0


def test_json_output_as_an_array_of_messages(fake):
    fake.mode("verbose_array", "left")
    assert ClaudeCLI(claude_bin=fake.bin).run("hi").text == "left"


def test_timeout_names_the_session_to_resume(fake):
    fake.mode("sleep")
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    t0 = time.monotonic()
    with pytest.raises(ModelTimeout) as ei:
        model.run("design", agentic=True, timeout_s=1.0, trace=tr)
    assert time.monotonic() - t0 < 10 and isinstance(ei.value, TimeoutError)
    assert ei.value.session_id == _flag(fake.calls()[0]["argv"], "--session-id")
    assert tr.records[-1]["ok"] is False and "timeout" in tr.records[-1]["error"]
    with pytest.raises(ModelTimeout) as ei:
        model.run("fix", agentic=True, session_id="abc", timeout_s=1.0)
    assert ei.value.session_id == "abc"


def test_missing_binary_and_cancellation_are_traced(fake, tmp_path):
    tr = TraceSink()
    with pytest.raises(OSError):
        ClaudeCLI(claude_bin=str(tmp_path / "no-such-claude")).run("hi", trace=tr)
    assert tr.records[-1]["ok"] is False
    fake.mode("sleep")
    with cancel_scope(lambda: "episode over"):
        with pytest.raises(ModelCallCancelled):
            ClaudeCLI(claude_bin=fake.bin).run("hi", trace=tr)
    assert tr.records[-1]["ok"] is False and "cancelled" in tr.records[-1]["error"]


def test_images_go_through_stream_json(fake, tmp_path):
    files = {"a.png": b"\x89PNG\r\n\x1a\nA", "b.jpg": b"\xff\xd8\xff\xe0B", "c.gif": b"GIF89aC",
             "d.webp": b"RIFF\x00\x00\x00\x00WEBPD", "e.jpeg": b"suffix decides"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    fake.mode("stream")
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    res = model.run("which way?", images=[str(tmp_path / n) for n in files], trace=tr)
    assert res.text == "image/png image/jpeg image/gif image/webp image/jpeg | which way?"
    call = fake.calls()[0]
    argv = call["argv"]
    assert _flag(argv, "--input-format") == "stream-json" and _flag(argv, "--output-format") == "stream-json"
    assert "--verbose" in argv and res.session_id == _flag(argv, "--session-id")
    line, = call["stdin"].splitlines()
    msg = json.loads(line)
    assert msg["type"] == "user" and msg["message"]["role"] == "user"
    assert msg["message"]["content"][0]["source"]["data"] == "iVBORw0KGgpB"      # base64 of a.png
    assert tr.records[-1]["request"]["images"] == [
        {"path": str((tmp_path / n).resolve()), "sha256": hashlib.sha256(d).hexdigest()} for n, d in files.items()]
    fake.mode("ok")
    model.run("text only")
    assert _flag(fake.calls()[1]["argv"], "--output-format") == "json"


@pytest.mark.parametrize("name,data,match", [
    ("missing.png", None, "cannot read image"),
    ("empty.png", b"", "empty"),
    ("frame.bmp", b"BM...", "png / jpeg / gif / webp"),
])
def test_unusable_images_are_traced_errors(fake, tmp_path, name, data, match):
    if data is not None:
        (tmp_path / name).write_bytes(data)
    tr = TraceSink()
    with pytest.raises(RuntimeError, match=match):
        ClaudeCLI(claude_bin=fake.bin).run("look", images=[str(tmp_path / name)], trace=tr)
    assert fake.calls() == [] and tr.records[-1]["ok"] is False
