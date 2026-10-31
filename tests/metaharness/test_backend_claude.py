"""claude backend through a fake ``claude`` CLI: player isolation, agentic restrictions, --effort,
quota/auth classification, traced failures, resumable timeouts, images via stream-json."""
from __future__ import annotations

import errno
import gc
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from metaharness.models import claude_cli as claude_mod
from metaharness.models import make_system2
from metaharness.models._cli import cancel_scope
from metaharness.models.base import (
    AGENT_PREAMBLE,
    PLAYER_PREAMBLE,
    ModelCallCancelled,
    ModelTimeout,
    ModelUnavailable,
    TraceSink,
)
from metaharness.models.claude_cli import EFFORT_LEVELS, ClaudeCLI

WATCHED_ENV = ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "TYPESAFE_API_KEY",
               "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

# Logs argv / cwd / stdin / env (and the --system-prompt-file it was given), then answers per FAKE_CLAUDE_MODE.
FAKE_CLAUDE = """
import json, os, sys, time
argv = sys.argv[1:]
stdin = sys.stdin.read()
system_file = argv[argv.index("--system-prompt-file") + 1] if "--system-prompt-file" in argv else None
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "stdin": stdin,
                         "system_file": open(system_file, encoding="utf-8").read() if system_file else None,
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
elif mode == "error_once":
    marker = os.environ["FAKE_CLAUDE_LOG"] + ".err"
    if not os.path.exists(marker):
        open(marker, "w").close()
        print(json.dumps({**result, "is_error": True, "result": text}))
        sys.exit(1)
    print(json.dumps({**result, "result": "up"}))
elif mode == "toolless_once":
    marker = os.environ["FAKE_CLAUDE_LOG"] + ".toolless"
    first = not os.path.exists(marker)
    open(marker, "w").close()
    print(json.dumps({**result, "num_turns": 1 if first else 4, "result": "no access" if first else "done"}))
elif mode == "error_then_answer":
    # the CLI's transcript: the first answer follows an API error; the resumed turn uses a tool
    store = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "projects", "p")
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, sid + ".jsonl"), "a") as fh:
        rows = ([{"type": "user", "message": {"content": stdin}}, {"type": "system", "subtype": "api_error"},
                 {"type": "assistant", "message": {"content": [{"type": "text", "text": "no files here"}]}}]
                if "--resume" not in argv else
                [{"type": "user", "message": {"content": stdin}},
                 {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
                 {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
                 {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}])
        fh.write("".join(json.dumps(r) + "\\n" for r in rows))
    print(json.dumps({**result, "num_turns": 3, "result": "done" if "--resume" in argv else "no files here"}))
elif mode == "rate_limit_once":
    marker = os.environ["FAKE_CLAUDE_LOG"] + ".429"
    if not os.path.exists(marker):
        open(marker, "w").close()
        print(json.dumps({**result, "is_error": True, "result": "API Error: 429 user rate limit exceeded (rpm)"}))
        sys.exit(1)
    print(json.dumps(result))
""" % {"watched": WATCHED_ENV}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE}")
    script.chmod(0o755)
    log = tmp_path / "claude-calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setattr(claude_mod, "_sleep", lambda s: None)          # 429 retries wait for nothing here
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))

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
    assert _flag(c1["argv"], "--permission-mode") == "bypassPermissions"
    # file / shell tools only: no subagents, skills, MCP servers or cross-session tools; web only when asked
    assert _flag(c1["argv"], "--tools") == "Bash,Read,Write,Edit,Glob,Grep,NotebookEdit"
    assert _flag(c2["argv"], "--tools").endswith(",WebSearch,WebFetch")
    assert "--strict-mcp-config" in c1["argv"] and "--disable-slash-commands" in c1["argv"]
    assert c1["argv"][c1["argv"].index("--disallowedTools") + 1:][:2] == ["WebSearch", "WebFetch"]
    assert "--disallowedTools" not in c2["argv"]
    assert _flag(c1["argv"], "--system-prompt") == f"{AGENT_PREAMBLE}\n\nSYS" and "--system-prompt" not in c3["argv"]
    assert c1["cwd"] == str(tmp_path.resolve())
    # agents keep their tools, but neither CLAUDE.md, auto-memory nor hooks may leak into the run directory
    assert c1["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and c1["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert json.loads(_flag(c1["argv"], "--settings")) == {"disableAllHooks": True}
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
    # exhausted credits, disabled usage and auth, as the 2.1.309 binary words them
    "You're out of usage credits. Switch to another model to continue.",
    "You're out of usage credits · resets 5pm (Europe/Berlin)",
    "Your org is out of usage · add funds to continue",
    "Your org is out of usage · contact your admin",
    "Your organization is out of usage credits. Contact your admin to add more.",
    "Your seat type doesn't include usage credits",
    "Your seat type doesn't include usage",
    "Your usage allocation has been disabled by your admin · ask your admin for a higher limit",
    "This service is disabled for your org",
    "Your group's usage limit is set to $0 · ask your admin for a higher limit",
    "You've hit your team's shared budget. Switch to another model to continue.",
    "You've hit your monthly spend limit. Switch to another model to continue.",
    "Your ANTHROPIC_API_KEY belongs to a disabled organization · Update or unset the environment variable",
    "Your organization has disabled API key authentication · Unset ANTHROPIC_API_KEY to use your claude.ai "
    "account instead",
    "Your organization has disabled Claude subscription access for Claude Code · Use an Anthropic API key instead, "
    "or ask your admin to enable access",
    "Anthropic profile login expired · Re-authenticate your Anthropic profile",
]
TRANSIENT = [
    'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
    "API Error: 500 Internal server error",
    "API Error: Request timed out.",
    "Repeated 529 Overloaded errors",
    "Opus is experiencing high load, please use /model to switch to Sonnet",
    "API Error: Request rejected (429) · this may be a temporary capacity issue.",
    "Usage credits are required for long context requests.",      # a shorter request still gets through
    # name a limit or a login, but the CLI itself calls them temporary
    "API Error: Server is temporarily limiting requests (not your usage limit) · this may be a temporary "
    "capacity issue.",
    "Another Claude Code process is refreshing the OAuth token. Retry in a moment.",
    "Could not refresh your login because another Claude Code process is refreshing it (or exited mid-refresh) · "
    "Try again in a minute; if it keeps happening, close other Claude Code windows or sign in again with /login",
    "Failed to refresh OAuth token: another Claude Code process is refreshing it or exited mid-refresh. This is "
    "usually transient; retry in a minute, and if it persists close other Claude Code processes or sign in again",
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


@pytest.mark.parametrize("mode", ["error", "errors", "stderr"])
@pytest.mark.parametrize("message", TRANSIENT)
def test_overload_and_server_errors_stay_runtime_errors(fake, mode, message):
    fake.mode(mode, message)
    with pytest.raises(RuntimeError) as ei:
        ClaudeCLI(claude_bin=fake.bin).run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


def test_an_agent_may_end_without_a_final_message_but_a_player_may_not(fake, tmp_path):
    fake.mode("empty")
    assert ClaudeCLI(claude_bin=fake.bin).run("work", cwd=tmp_path, agentic=True).text == ""
    with pytest.raises(RuntimeError, match="empty result"):
        ClaudeCLI(claude_bin=fake.bin).run("move?")


def test_rate_limits_are_retried_and_agents_resume_their_session(fake, tmp_path):
    fake.mode("rate_limit_once")
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    assert model.run("hi", trace=tr).text == "up"                        # a player repeats the request
    c1, c2 = fake.calls()
    assert _flag(c1["argv"], "--session-id") != _flag(c2["argv"], "--session-id") and c2["stdin"] == "hi"
    assert "transient failure (retry 1" in tr.records[0]["error"] and tr.records[1]["ok"]
    (tmp_path / "claude-calls.jsonl.429").unlink()
    store = tmp_path / "claude-config" / "projects" / "some-dir"
    store.mkdir(parents=True)
    (store / "abc.jsonl").write_text("{}")                                # the CLI stored the agent's session
    assert model.run("design", cwd=tmp_path, agentic=True, session_id="abc").text == "up"
    c3, c4 = fake.calls()[2:]
    assert _flag(c3["argv"], "--resume") == "abc" and _flag(c4["argv"], "--resume") == "abc"
    assert c4["stdin"].startswith("The previous turn was interrupted by a transient API error")
    fake.mode("error", "API Error: 429 user rate limit exceeded (rpm)")
    with pytest.raises(RuntimeError, match="after 5 retries") as ei:   # never unavailable
        model.run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


def test_a_new_agent_session_without_tool_use_is_retried_once_in_a_new_session(fake, tmp_path):
    fake.mode("toolless_once")
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    assert model.run("design", cwd=tmp_path, agentic=True, trace=tr).text == "done"
    c1, c2 = fake.calls()
    assert _flag(c1["argv"], "--session-id") != _flag(c2["argv"], "--session-id") and c2["stdin"] == "design"
    assert "without using any tool" in tr.records[0]["error"] and tr.records[1]["ok"]
    marker = tmp_path / "claude-calls.jsonl.toolless"
    for kw in ({"agentic": True, "session_id": "abc"}, {}):            # a resumed session, a player: as answered
        marker.unlink()
        assert model.run("go", cwd=tmp_path, **kw).text == "no access"
    assert len(fake.calls()) == 4


def test_an_answer_right_after_an_api_error_is_followed_by_a_resume(fake, tmp_path):
    fake.mode("error_then_answer")
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    assert model.run("build it", cwd=tmp_path, agentic=True, trace=tr).text == "done"
    c1, c2 = fake.calls()
    assert _flag(c2["argv"], "--resume") == _flag(c1["argv"], "--session-id")
    assert c2["stdin"].startswith("Your previous answer came right after a transient API error")
    assert "right after an API error" in tr.records[0]["error"] and tr.records[1]["ok"]
    assert len(fake.calls()) == 2                                   # the resumed turn used a tool: done


def test_a_temporary_line_does_not_hide_an_exhausted_account(fake):
    fake.mode("error", "Another Claude Code process is refreshing the OAuth token. Retry in a moment.\n"
                       "You're out of usage credits. Switch to another model to continue.")
    with pytest.raises(ModelUnavailable, match="matched 'out of usage'"):
        ClaudeCLI(claude_bin=fake.bin).run("hi")


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
    with pytest.raises(ModelUnavailable, match="could not run: FileNotFoundError") as ei:
        ClaudeCLI(claude_bin=str(tmp_path / "no-such-claude")).run("hi", trace=tr)
    # a CLI that cannot start makes every call fail: the run stops instead of burning its iterations
    assert not isinstance(ei.value, OSError)
    assert tr.records[-1]["ok"] is False and "FileNotFoundError" in tr.records[-1]["error"]
    fake.mode("sleep")
    with cancel_scope(lambda: "episode over"):
        with pytest.raises(ModelCallCancelled):
            ClaudeCLI(claude_bin=fake.bin).run("hi", trace=tr)
    assert tr.records[-1]["ok"] is False and "cancelled" in tr.records[-1]["error"]


def test_an_argv_the_os_refuses_is_a_traced_runtime_error(monkeypatch):
    def run_cli(argv, **kw):
        raise OSError(errno.E2BIG, os.strerror(errno.E2BIG), argv[0])
    monkeypatch.setattr(claude_mod, "run_cli", run_cli)
    tr = TraceSink()
    with pytest.raises(RuntimeError, match=f"could not run: OSError: .*{os.strerror(errno.E2BIG)}") as ei:
        ClaudeCLI().run("hi", trace=tr)
    assert not isinstance(ei.value, OSError) and isinstance(ei.value.__cause__, OSError)
    assert tr.records[-1]["ok"] is False and os.strerror(errno.E2BIG) in tr.records[-1]["error"]


def test_long_system_prompts_go_through_a_file(fake, tmp_path):
    """One argv element over 128 KiB fails with E2BIG on Linux: a long system prompt reaches the CLI as
    --system-prompt-file (the player's in its private directory, an agent's never in its workspace),
    removed after the call, timed out or not; a short one stays on argv."""
    rules = "règle · " * 40_000                   # 400 KB of UTF-8
    player_text = f"{PLAYER_PREAMBLE}\n\n{rules}"
    model, tr = ClaudeCLI(claude_bin=fake.bin), TraceSink()
    first = model.run("state", system=rules, trace=tr)
    model.run("next", session_id=first.session_id, system=rules)
    model.run("design", cwd=tmp_path, agentic=True, system=rules)
    model.run("short", system="rules")
    fake.mode("sleep")
    with pytest.raises(ModelTimeout):
        model.run("slow", system=rules, timeout_s=1.0)
    c1, c2, c3, c4, c5 = fake.calls()
    for call, text in ((c1, player_text), (c2, player_text), (c3, f"{AGENT_PREAMBLE}\n\n{rules}"), (c5, player_text)):
        argv = call["argv"]
        assert "--system-prompt" not in argv and call["system_file"] == text
        assert max(len(a.encode()) for a in argv) < 4096
        path = Path(_flag(argv, "--system-prompt-file"))
        assert path.is_absolute() and not path.exists()
    assert Path(_flag(c1["argv"], "--system-prompt-file")).parent == Path(c1["cwd"])
    assert tmp_path.resolve() not in Path(_flag(c3["argv"], "--system-prompt-file")).parents
    assert _flag(c4["argv"], "--system-prompt") == f"{PLAYER_PREAMBLE}\n\nrules" and c4["system_file"] is None
    assert "--system-prompt-file" not in c4["argv"] and not any(Path(c1["cwd"]).iterdir())
    assert tr.records[0]["request"]["system"] == player_text and tr.records[0]["ok"] is True


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


@pytest.mark.parametrize("message", ["API Error: Responses stream error: Upstream request failed (upstream_error)",
                                     'API Error: 529 {"type":"error","error":{"type":"overloaded_error"}}',
                                     "API Error: 500 Internal server error"])
def test_transient_api_failures_are_retried(fake, message):
    fake.mode("error_once", message)
    assert ClaudeCLI(claude_bin=fake.bin).run("hi").text == "up" and len(fake.calls()) == 2


def test_quota_failures_are_not_retried(fake):
    fake.mode("error", "API Error: 503 · You've hit your usage limit")
    with pytest.raises(ModelUnavailable):
        ClaudeCLI(claude_bin=fake.bin).run("hi")
    assert len(fake.calls()) == 1


def test_slot_waiters_are_served_in_arrival_order(tmp_path, monkeypatch):
    import subprocess
    import threading
    monkeypatch.setattr(claude_mod, "_SLOT_DIR", tmp_path / "slots")
    monkeypatch.setenv("METAHARNESS_CLAUDE_AGENT_SLOTS", "1")
    queue = tmp_path / "slots" / "agent.queue"
    queue.mkdir(parents=True)
    dead = subprocess.Popen(["true"])
    dead.wait()
    (queue / f"{1:020d}-{dead.pid}-deadbeef").touch()          # a ticket left by a process that died
    order: list[str] = []
    release = threading.Event()

    def worker(name: str) -> None:
        with claude_mod._slot("agent"):
            order.append(name)
            if name == "first":
                release.wait(10)

    first = threading.Thread(target=worker, args=("first",))
    first.start()
    while not order:
        time.sleep(0.01)
    threads = []
    for name in ("second", "third", "fourth"):                  # queue up in this order behind "first"
        threads.append(threading.Thread(target=worker, args=(name,)))
        threads[-1].start()
        while len(os.listdir(queue)) < len(threads):
            time.sleep(0.01)
    release.set()
    for th in [first, *threads]:
        th.join(10)
    assert order == ["first", "second", "third", "fourth"] and os.listdir(queue) == []


def test_an_agents_time_starts_when_it_gets_a_slot(fake, tmp_path, monkeypatch):
    import contextlib as _cl
    waits = []

    @_cl.contextmanager
    def slow_slot(pool, since=None):
        time.sleep(1.5)                     # another process held the slot
        waits.append(pool)
        yield

    monkeypatch.setattr(claude_mod, "_slot", slow_slot)
    fake.mode("ok")
    assert ClaudeCLI(claude_bin=fake.bin).run("work", cwd=tmp_path, agentic=True, timeout_s=1.0).text == "up"
    with pytest.raises(ModelTimeout):        # a player's wait is part of its latency: it times out
        ClaudeCLI(claude_bin=fake.bin).run("move?", timeout_s=1.0)
    assert waits == ["agent", "player"]
