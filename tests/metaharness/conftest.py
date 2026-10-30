import pytest


@pytest.fixture
def tmp_runs(tmp_path):
    return tmp_path / "runs"


@pytest.fixture(autouse=True)
def _no_shared_claude_slots(monkeypatch):
    """The claude backend's cross-process slots are shared with live searches on the machine: tests must
    never wait for (or hold) them."""
    monkeypatch.setenv("METAHARNESS_CLAUDE_AGENT_SLOTS", "0")
    monkeypatch.setenv("METAHARNESS_CLAUDE_PLAYER_SLOTS", "0")
    monkeypatch.delenv("METAHARNESS_CLAUDE_BASE_URL", raising=False)
