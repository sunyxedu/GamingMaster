import pytest


@pytest.fixture
def tmp_runs(tmp_path):
    return tmp_path / "runs"
