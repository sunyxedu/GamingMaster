import os

from metaharness.config import load_dotenv, require_env


def test_load_dotenv_parses_and_does_not_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nFOO_TEST=bar # trailing\nQUOTED='x y'\nEMPTY=\n")
    monkeypatch.setenv("FOO_TEST", "already")
    monkeypatch.delenv("QUOTED", raising=False)
    got = load_dotenv(p)
    assert got == {"FOO_TEST": "bar", "QUOTED": "x y", "EMPTY": ""}
    assert os.environ["FOO_TEST"] == "already"
    assert os.environ["QUOTED"] == "x y"
    load_dotenv(p, override=True)
    assert os.environ["FOO_TEST"] == "bar"


def test_require_env_message(monkeypatch):
    monkeypatch.delenv("DEFINITELY_MISSING_KEY", raising=False)
    try:
        require_env("DEFINITELY_MISSING_KEY")
    except RuntimeError as e:
        assert ".env" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_env_file_is_gitignored():
    from metaharness.config import REPO_ROOT
    ignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore
