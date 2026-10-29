"""codex backend: prompt through stdin, structural failure classification, per-call usage of
resumed threads, timeouts that keep the thread id, image provenance, player isolation."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from metaharness.models import codex_cli as codex_mod
from metaharness.models.base import PLAYER_PREAMBLE, ModelTimeout, ModelUnavailable, TraceSink
from metaharness.models.codex_cli import CodexCLI

# Logs what it got, then replays a turn (FAKE_CODEX_MODE=echo) or hangs after starting one (sleep).
FAKE_CODEX = """
import json, os, sys, time
argv = sys.argv[1:]
prompt = sys.stdin.read() if argv[-1] == "-" else argv[-1]
with open(os.environ["FAKE_CODEX_LOG"], "a") as fh:
    fh.write(json.dumps({"argv": argv, "stdin": prompt}) + "\\n")
thread = argv[argv.index("resume") + 1] if "resume" in argv else "t-new"
print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)
if os.environ.get("FAKE_CODEX_MODE") == "sleep":
    time.sleep(30)
text = f"got {len(prompt)} chars"
open(argv[argv.index("-o") + 1], "w").write(text)
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 1}}))
"""


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    script.chmod(0o755)
    log = tmp_path / "codex-calls.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    return SimpleNamespace(bin=str(script), calls=lambda: [
        json.loads(line) for line in log.read_text().splitlines()] if log.exists() else [])


def _events(*events: dict) -> str:
    return "".join(json.dumps(ev) + "\n" for ev in events)


def _answer(text: str, thread: str = "t-1", usage: dict | None = None) -> str:
    return _events({"type": "thread.started", "thread_id": thread},
                   {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
                   {"type": "turn.completed", "usage": usage or {"input_tokens": 3, "output_tokens": 1}})


def _fake_run_cli(stdout: str, rc: int = 0, stderr: str = "", seen: list | None = None):
    def run_cli(argv, *, cwd, stdin_text, timeout_s, env_overrides=None, env_drop=()):
        if seen is not None:
            seen.append({"argv": list(argv), "stdin": stdin_text, "env_drop": tuple(env_drop),
                         "env_overrides": env_overrides})
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)
    return run_cli


def test_large_prompt_reaches_codex_through_stdin(fake_codex):
    model = CodexCLI(model="m", codex_bin=fake_codex.bin)
    prompt = "x" * 300_000           # a single argv element >= 128 KiB fails with E2BIG on Linux
    res = model.run(prompt)
    full = PLAYER_PREAMBLE + "\n\n" + prompt
    assert res.text == f"got {len(full)} chars" and res.session_id == "t-new"
    resumed = model.run("y" * 300_000, session_id="t-new", agentic=True)
    assert resumed.session_id == "t-new"
    first, second = fake_codex.calls()
    assert first["stdin"] == full and second["stdin"] == "y" * 300_000
    for call in (first, second):
        assert call["argv"][-1] == "-" and max(len(a) for a in call["argv"]) < 4096
    assert second["argv"][:3] == ["exec", "resume", "t-new"]


def test_answer_mentioning_limits_or_401_is_a_success(monkeypatch):
    text = "You've hit your usage limit? No: the snake is unauthorized to pass 401 walls. Answer: up"
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(_answer(text), stderr="usage limit 401 in a log"))
    assert CodexCLI(model="m").run("hi").text == text


def test_error_event_inside_a_completed_turn_is_a_warning(monkeypatch):
    retry = "stream disconnected - retrying sampling request (1/5 in 200ms)..."
    stdout = _events({"type": "error", "message": retry}) + _answer("up")
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(stdout))
    tr = TraceSink()
    assert CodexCLI(model="m").run("hi", trace=tr).text == "up"
    assert tr.records[-1]["ok"] is True and tr.records[-1]["response"]["warnings"] == [retry]


def test_timestamp_containing_401_is_not_an_auth_failure(monkeypatch):
    stdout = _events({"type": "turn.failed", "error": {"message": "stream error at 2026-10-22T10:40:14.401Z (id 14012)"}})
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(stdout, rc=1))
    with pytest.raises(RuntimeError) as ei:
        CodexCLI(model="m").run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


@pytest.mark.parametrize("message,what", [
    ("unexpected status 401 Unauthorized", "auth problem"),
    ("Your access token could not be refreshed. Please log out and sign in again.", "auth problem"),
    ("You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro)", "usage limit"),
    ("Quota exceeded. Check your plan and billing details.", "usage limit"),
    # UsageLimitReachedError of codex 0.160 for workspace credits and spend caps
    ("Your workspace is out of credits. Add credits to continue.", "usage limit"),
    ("Your workspace is out of credits. Ask your workspace owner to refill in order to continue.", "usage limit"),
    ("You hit your spend cap set in your workspace. Increase your spend cap to continue.", "usage limit"),
    ("You hit your spend cap set by the owner of your workspace. Ask an owner to increase your spend cap to "
     "continue.", "usage limit"),
])
def test_reported_failures_are_classified_with_the_matched_text(monkeypatch, message, what):
    stdout = _events({"type": "thread.started", "thread_id": "t"}, {"type": "error", "message": message},
                     {"type": "turn.failed", "error": {"message": message}})
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(stdout, rc=1))
    tr = TraceSink()
    with pytest.raises(ModelUnavailable, match=f"codex {what} \\(matched ") as ei:
        CodexCLI(model="m").run("hi", trace=tr)
    assert message[:30] in str(ei.value) and tr.records[-1]["ok"] is False


@pytest.mark.parametrize("message", [
    "Selected model is at capacity. Please try a different model.",
    "We’re currently experiencing high demand, which may cause temporary errors.",
    "Codex is currently experiencing high load.",
    "unexpected status 503 Service Unavailable: upstream connect error",
])
def test_overload_stays_a_runtime_error(monkeypatch, message):
    stdout = _events({"type": "thread.started", "thread_id": "t"}, {"type": "error", "message": message},
                     {"type": "turn.failed", "error": {"message": message}})
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(stdout, rc=1))
    with pytest.raises(RuntimeError, match="codex exec failed") as ei:
        CodexCLI(model="m").run("hi")
    assert not isinstance(ei.value, ModelUnavailable)


def test_a_codex_that_cannot_start_is_a_traced_runtime_error(tmp_path):
    tr = TraceSink()
    with pytest.raises(RuntimeError, match="could not run: FileNotFoundError") as ei:
        CodexCLI(model="m", codex_bin=str(tmp_path / "no-such-codex")).run("hi", trace=tr)
    assert not isinstance(ei.value, OSError)                 # harnesses catch RuntimeError, not OSError
    assert tr.records[-1]["ok"] is False and "FileNotFoundError" in tr.records[-1]["error"]


def test_stderr_is_classified_only_when_codex_failed(monkeypatch):
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli("", rc=1, stderr="Error: Not logged in"))
    with pytest.raises(ModelUnavailable, match="auth problem"):
        CodexCLI(model="m").run("hi")
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli("", rc=2, stderr="error: unexpected argument '--bogus'"))
    with pytest.raises(RuntimeError, match="rc=2.*bogus"):
        CodexCLI(model="m").run("hi")


def test_resumed_threads_report_per_call_usage(monkeypatch):
    """turn.completed carries the thread's running total (archived traces: 12724 -> 27213 -> 42212 ...)."""
    totals = iter([{"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10},
                   {"input_tokens": 250, "cached_input_tokens": 90, "output_tokens": 25},
                   {"input_tokens": 420, "cached_input_tokens": 230, "output_tokens": 40}])

    def run_cli(argv, *, cwd, stdin_text, timeout_s, env_overrides=None, env_drop=()):
        return subprocess.CompletedProcess(argv, 0, _answer("up", "t-1", next(totals)), "")
    monkeypatch.setattr(codex_mod, "run_cli", run_cli)
    model, tr = CodexCLI(model="m"), TraceSink()
    first = model.run("a", trace=tr)
    second = model.run("b", session_id=first.session_id, trace=tr)
    third = model.run("c", session_id=second.session_id, trace=tr)
    assert first.usage["input_tokens"] == 100 and second.usage["input_tokens"] == 150
    assert third.usage == {"input_tokens": 170, "cached_input_tokens": 140, "output_tokens": 15,
                           "thread_cumulative": {"input_tokens": 420, "cached_input_tokens": 230, "output_tokens": 40}}
    assert [r["usage"]["input_tokens"] for r in tr.records] == [100, 150, 170]
    assert model.total_usage == {"input_tokens": 420, "cached_input_tokens": 230, "output_tokens": 40}
    assert model.n_calls == 3


def test_resuming_an_unseen_thread_flags_the_upper_bound(monkeypatch):
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(_answer("up", "t-old", {"input_tokens": 900})))
    res = CodexCLI(model="m").run("hi", session_id="t-old")
    assert res.usage["input_tokens"] == 900 and res.usage["thread_baseline"] == "unknown"


def test_usage_accounting_is_thread_safe(monkeypatch):
    def run_cli(argv, *, cwd, stdin_text, timeout_s, env_overrides=None, env_drop=()):
        thread = argv[argv.index("resume") + 1] if "resume" in argv else stdin_text.rsplit(" ", 1)[-1]
        n = int(stdin_text.split()[-2]) if "resume" in argv else 1
        return subprocess.CompletedProcess(argv, 0, _answer("up", thread, {"input_tokens": 10 * n}), "")
    monkeypatch.setattr(codex_mod, "run_cli", run_cli)
    model = CodexCLI(model="m")

    def player(i: int) -> None:
        sid = model.run(f"start t{i}", agentic=True).session_id
        for n in range(2, 6):
            sid = model.run(f"turn {n} {sid}", session_id=sid, agentic=True).session_id
    workers = [threading.Thread(target=player, args=(i,)) for i in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert model.total_usage == {"input_tokens": 8 * 50} and model.n_calls == 40


def test_timeout_names_the_thread_to_resume(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    model, tr = CodexCLI(model="m", codex_bin=fake_codex.bin), TraceSink()
    t0 = time.monotonic()
    with pytest.raises(ModelTimeout) as ei:
        model.run("design a harness", agentic=True, timeout_s=1.5, trace=tr)
    assert isinstance(ei.value, TimeoutError) and ei.value.session_id == "t-new"
    assert time.monotonic() - t0 < 10
    assert tr.records[-1]["ok"] is False and "timeout" in tr.records[-1]["error"]
    with pytest.raises(ModelTimeout) as ei:
        model.run("fix it", session_id="t-7", agentic=True, timeout_s=1.5)
    assert ei.value.session_id == "t-7"


def test_images_are_attached_by_absolute_path_and_traced(tmp_path, monkeypatch):
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nframe")
    seen: list = []
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(_answer("left"), seen=seen))
    monkeypatch.chdir(tmp_path)
    tr = TraceSink()
    assert CodexCLI(model="m").run("look", images=["frame.png"], trace=tr).text == "left"
    argv = seen[0]["argv"]
    assert argv[argv.index("-i") + 1] == str(img.resolve())
    assert tr.records[-1]["request"]["images"] == [
        {"path": str(img.resolve()), "sha256": hashlib.sha256(img.read_bytes()).hexdigest()}]
    with pytest.raises(RuntimeError, match="cannot read image"):
        CodexCLI(model="m").run("look", images=[str(tmp_path / "missing.png")], trace=tr)
    assert len(seen) == 1 and tr.records[-1]["ok"] is False


def test_player_turns_off_file_and_hook_features(tmp_path):
    a = CodexCLI(model="m")
    player = a._argv(cwd=None, session_id=None, agentic=False, last_msg_path=tmp_path / "l")
    agent = a._argv(cwd=tmp_path, session_id=None, agentic=True, last_msg_path=tmp_path / "l")
    for feature in ("view_image", "image_generation", "plugins", "hooks", "memories"):
        assert f"features.{feature}=false" in player and f"features.{feature}=false" not in agent


def test_player_prompt_and_env(monkeypatch):
    seen: list = []
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(_answer("up"), seen=seen))
    CodexCLI(model="m").run("state", system="rules")
    assert seen[0]["stdin"] == f"{PLAYER_PREAMBLE}\n\nrules\n\nstate"
    assert {"TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"} <= set(seen[0]["env_drop"])


def test_codex_binary_that_cannot_start_is_unavailable(tmp_path):
    from metaharness.models.base import ModelUnavailable
    with pytest.raises(ModelUnavailable, match="could not run"):
        CodexCLI(model="m", codex_bin=str(tmp_path / "no-such-codex")).run("hi")


# ---------------------------------------------------------------- providers (OpenAI-compatible endpoint)

ROUTER = {"id": "router", "name": "r", "base_url": "http://127.0.0.1:8787/v1", "env_key": "MH_ROUTER_API_KEY",
          "wire_api": "responses"}


def test_provider_profile_runs_with_overrides_and_its_own_codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_ROUTER_API_KEY", "secret")
    seen: list = []
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(_answer("up"), seen=seen))
    home = tmp_path / "codex-home"
    model = CodexCLI(model="gpt-6-astra", provider={**ROUTER, "codex_home": str(home)})
    assert model.name == "codex:gpt-6-astra@router"
    assert model.run("state").text == "up"
    argv = seen[0]["argv"]
    cfg = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in argv if "=" in a}
    assert cfg["model_provider"] == '"router"'
    assert cfg["model_providers.router.base_url"] == '"http://127.0.0.1:8787/v1"'
    assert cfg["model_providers.router.env_key"] == '"MH_ROUTER_API_KEY"'
    assert cfg["model_providers.router.wire_api"] == '"responses"'
    assert "shell_environment_policy.exclude" not in cfg          # a player has no shell
    assert seen[0]["env_overrides"] == {"CODEX_HOME": str(home)}
    toml = (home / "config.toml").read_text()                     # reproducible by hand: CODEX_HOME=<dir> codex
    assert 'model_provider = "router"' in toml and "[model_providers.router]" in toml and "secret" not in toml
    assert "MH_ROUTER_API_KEY" not in seen[0]["env_drop"]         # codex itself must see the key


def test_provider_key_is_hidden_from_the_agent_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_ROUTER_API_KEY", "secret")
    model = CodexCLI(model="m", provider={**ROUTER, "codex_home": str(tmp_path)})
    agent = model._argv(cwd=tmp_path, session_id=None, agentic=True, last_msg_path=tmp_path / "l")
    assert 'shell_environment_policy.exclude=["MH_ROUTER_API_KEY"]' in agent
    resumed = model._argv(cwd=tmp_path, session_id="t", agentic=True, last_msg_path=tmp_path / "l")
    assert 'model_provider="router"' in resumed                   # resume gets the provider too


def test_provider_without_its_key_is_refused_at_construction(monkeypatch):
    monkeypatch.delenv("MH_ROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MH_ROUTER_API_KEY.*not set"):
        CodexCLI(model="m", provider=ROUTER)
    with pytest.raises(ValueError, match="base_url"):
        CodexCLI(model="m", provider={"id": "x"})
    with pytest.raises(ValueError, match="provider id"):
        CodexCLI(model="m", provider={"id": "bad id", "base_url": "http://x"})


def test_profile_naming_an_unknown_provider_is_an_error(tmp_path, monkeypatch):
    from metaharness.models import make_system2
    cfg = tmp_path / "models.toml"
    cfg.write_text('[system2.x]\nbackend = "codex"\nmodel = "m"\nprovider = "nowhere"\n')
    monkeypatch.setenv("METAHARNESS_MODELS", str(cfg))
    with pytest.raises(ValueError, match="provider 'nowhere'"):
        make_system2("x")


def test_toml_values():
    from metaharness.models.codex_cli import _toml
    assert _toml("a") == '"a"' and _toml(True) == "true" and _toml(3) == "3"
    assert _toml(["k"]) == '["k"]' and _toml({"a": 1}) == "{a = 1}"


def test_rate_limited_call_is_retried_and_resumes_its_thread(monkeypatch):
    """codex gives up on 429 after its own retries; the backend waits and resumes the thread."""
    limited = _events({"type": "thread.started", "thread_id": "t-rl"},
                      {"type": "error", "message": "exceeded retry limit, last status: 429 Too Many Requests"},
                      {"type": "turn.failed", "error": {"message": "exceeded retry limit, last status: 429 Too Many Requests"}})
    outcomes = iter([(limited, 1), (limited, 1), (_answer("done", "t-rl"), 0)])
    seen: list = []

    def run_cli(argv, *, cwd, stdin_text, timeout_s, env_overrides=None, env_drop=()):
        seen.append({"argv": list(argv), "stdin": stdin_text})
        stdout, rc = next(outcomes)
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")
    monkeypatch.setattr(codex_mod, "run_cli", run_cli)
    naps: list = []
    monkeypatch.setattr(codex_mod.time, "sleep", naps.append)
    tr = TraceSink()
    res = CodexCLI(model="m").run("build it", agentic=True, trace=tr)
    assert res.text == "done" and len(seen) == 3 and len(naps) == 2 and naps[1] > naps[0] / 4
    assert seen[1]["argv"][:3] == ["codex", "exec", "resume"] and seen[1]["argv"][3] == "t-rl"
    assert "rate-limited" in seen[1]["stdin"] and seen[0]["stdin"] == "build it"
    assert [r["type"] for r in tr.records].count("rate_limited") == 2 and tr.records[-1]["ok"] is True
    # a player call is simply sent again (same prompt, no thread continuation text)
    outcomes = iter([(limited, 1), (_answer("up"), 0)])
    seen.clear()
    assert CodexCLI(model="m").run("state").text == "up" and seen[1]["stdin"] == seen[0]["stdin"]


def test_rate_limit_retries_give_up_after_the_wait_budget(monkeypatch):
    limited = _events({"type": "turn.failed", "error": {"message": "429 Too Many Requests"}})
    monkeypatch.setattr(codex_mod, "run_cli", _fake_run_cli(limited, rc=1))
    monkeypatch.setattr(codex_mod.time, "sleep", lambda s: None)
    model = CodexCLI(model="m")
    model.rate_limit_wait_s = 12.0            # 5 s + 10 s backoff: one retry fits, the second does not
    with pytest.raises(RuntimeError, match="429") as ei:
        model.run("hi")
    assert not isinstance(ei.value, ModelUnavailable) and type(ei.value) is RuntimeError


def test_concurrency_slots_are_shared_through_lock_files(tmp_path):
    from metaharness.models.codex_cli import concurrency_slot
    active, peak, lock = [0], [0], threading.Lock()

    def worker():
        with concurrency_slot(tmp_path / "slots", 2, poll_s=0.01):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
    ts = [threading.Thread(target=worker) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 2 and sorted(p.name for p in (tmp_path / "slots").iterdir()) == ["0.lock", "1.lock"]
