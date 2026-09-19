"""Tiny ``.env`` loader (no python-dotenv dependency) + project paths."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``.env`` into ``os.environ``.

    Never logs values. Existing environment wins unless ``override``.
    Returns the parsed mapping (for tests).
    """
    p = Path(path) if path is not None else REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def require_env(name: str) -> str:
    load_dotenv()
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Put it in {REPO_ROOT / '.env'} (see .env.example)."
        )
    return value
