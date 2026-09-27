"""PEP 517 build backend of gmsim: maturin, with a user-level rustup put on PATH
when ``cargo`` is not already there (``$CARGO_HOME/bin``, else ``~/.cargo/bin``) —
so ``uv sync`` builds the extension in shells that never sourced ``~/.cargo/env``
(cluster login nodes, IDEs, non-interactive shells).  Every PEP 517 hook is
maturin's own (``pyproject.toml``: ``build-backend = "gmsim_build"``,
``backend-path = ["."]``)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _add_user_cargo_to_path() -> None:
    if shutil.which("cargo"):
        return
    exe = "cargo.exe" if os.name == "nt" else "cargo"
    for home in (os.environ.get("CARGO_HOME"), str(Path.home() / ".cargo")):
        if home and (Path(home) / "bin" / exe).is_file():
            os.environ["PATH"] = str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", "")
            return


_add_user_cargo_to_path()

import maturin as _maturin  # noqa: E402  (after the PATH fix: maturin runs cargo)


def __getattr__(name: str):
    """Every PEP 517 hook (build_wheel, build_editable, prepare_metadata_*, ...)."""
    return getattr(_maturin, name)
