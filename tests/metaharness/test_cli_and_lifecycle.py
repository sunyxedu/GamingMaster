"""CLI subprocess plumbing (cancellation, deadlines, large stdin) and adaptor lifecycle helpers."""
from __future__ import annotations

import os
import sys
import time

import pytest

from metaharness.games.base import ActionSpec, close_game
from metaharness.models._cli import CLITimeout, cancel_scope, run_cli
from metaharness.models.base import ModelCallCancelled

PY = sys.executable


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a killed child may linger as a zombie until reaped; treat zombies as dead
    try:
        with open(f"/proc/{pid}/status") as fh:
            return "zombie" not in fh.read().lower()
    except OSError:
        return True


def test_run_cli_large_stdin_and_output():
    payload = "x" * 300_000
    proc = run_cli([PY, "-c", "import sys; d = sys.stdin.read(); print(len(d))"],
                   cwd=None, stdin_text=payload, timeout_s=30)
    assert proc.returncode == 0 and proc.stdout.strip() == "300000"


def test_run_cli_timeout_keeps_partial_output():
    t0 = time.monotonic()
    with pytest.raises(CLITimeout) as ei:
        run_cli([PY, "-u", "-c", "import time; print('partial', flush=True); time.sleep(30)"],
                cwd=None, stdin_text=None, timeout_s=1.0)
    assert time.monotonic() - t0 < 10
    assert isinstance(ei.value, TimeoutError) and "partial" in ei.value.stdout


def test_run_cli_cancel_scope_kills_process(tmp_path):
    pid_file = tmp_path / "pid"
    started = time.monotonic()
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    with cancel_scope(lambda: "episode over" if time.monotonic() - started > 0.5 else None):
        with pytest.raises(ModelCallCancelled):
            run_cli([PY, "-c", code], cwd=None, stdin_text=None, timeout_s=30)
    assert time.monotonic() - started < 5
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_run_cli_without_scope_is_not_polled():
    # outside a cancel scope a long call simply completes (proposer / study / adapt agents)
    with cancel_scope(lambda: None):
        pass
    proc = run_cli([PY, "-c", "import time; time.sleep(0.5); print('ok')"], cwd=None, stdin_text=None, timeout_s=10)
    assert proc.stdout.strip() == "ok"


def test_run_cli_interrupt_kills_child(tmp_path):
    pid_file = tmp_path / "pid"
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"

    def interrupt():
        if pid_file.exists():
            raise KeyboardInterrupt
        return None

    with cancel_scope(interrupt):
        with pytest.raises(KeyboardInterrupt):
            run_cli([PY, "-c", code], cwd=None, stdin_text=None, timeout_s=30)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_close_game_optional_and_bounded():
    class NoClose:
        pass

    class Closes:
        closed = 0

        def close(self):
            Closes.closed += 1

    class Raises:
        def close(self):
            raise RuntimeError("bridge gone")

    class Hangs:
        def close(self):
            time.sleep(5)

    assert close_game(NoClose()) is None
    assert close_game(Closes()) is None and Closes.closed == 1
    assert "bridge gone" in close_game(Raises())
    t0 = time.monotonic()
    assert "did not return" in close_game(Hangs(), timeout_s=0.2)
    assert time.monotonic() - t0 < 2


def test_action_spec_describes_sticky_mailbox():
    text = ActionSpec(names=["a", "b"], noop="b").describe()
    assert "repeats the last posted action" in text


def test_run_cli_delivers_large_stdin_inside_a_cancel_scope():
    # the child starts slowly (like a CLI loading its runtime) and only then reads stdin: a polled
    # communicate() used to drop everything beyond the pipe buffer and time out
    payload = "y" * 300_000
    code = "import sys, time; time.sleep(0.5); print(len(sys.stdin.read()))"
    with cancel_scope(lambda: None):
        proc = run_cli([PY, "-c", code], cwd=None, stdin_text=payload, timeout_s=20)
    assert proc.returncode == 0 and proc.stdout.strip() == "300000"


def test_kill_all_kills_live_cli_children(tmp_path):
    import threading
    from metaharness.models import _cli
    pid_file = tmp_path / "pid"
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    errors = []

    def call():
        try:
            run_cli([PY, "-c", code], cwd=None, stdin_text=None, timeout_s=30)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    th = threading.Thread(target=call, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    _cli.kill_all()
    th.join(5)
    pid = int(pid_file.read_text())
    assert not th.is_alive() and not _alive(pid)


def test_a_call_cancelled_at_its_own_deadline_is_cancelled_not_timed_out():
    # the episode closes a moment before the CLI's (clamped) timeout: the cancel must win
    t0 = time.monotonic()
    with cancel_scope(lambda: "episode deadline" if time.monotonic() - t0 > 0.95 else None):
        with pytest.raises(ModelCallCancelled):
            run_cli([PY, "-c", "import time; time.sleep(30)"], cwd=None, stdin_text="x" * 10, timeout_s=1.0)
