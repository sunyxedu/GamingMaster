#!/usr/bin/env python3
"""generals.io data pipeline — thin wrapper around the Rust implementation.

    uv run python -m scripts.prepare_data --players Human.exe --output data/output \\
        --target 1000 --min-turns 80

Same flags, layout and manifest semantics as before (fetch metadata → filter →
download → simulate → convert, resumable, sharded with --shard k/n, --merge);
the work is done by ``gmdata generals pipeline`` (data_processing/rs/gmdata),
built on first use with cargo. For strong-human ladder data use
``gmdata generals ladder`` instead (see data_processing/README.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "data_processing"
BINARY = WORKSPACE / "target" / "release" / "gmdata"


def gmdata() -> Path:
    """Path to the gmdata binary, building it when missing (needs cargo)."""
    if BINARY.exists():
        return BINARY
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        sys.exit("gmdata is not built and cargo was not found: install Rust (https://rustup.rs) "
                 "and run `cargo build --release -p gmdata` in data_processing/")
    subprocess.run([cargo, "build", "--release", "-p", "gmdata"], cwd=WORKSPACE, check=True)
    return BINARY


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--players", "-p", nargs="+", required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--target", "--max-replays", type=int, default=10000)
    p.add_argument("--min-turns", type=int, default=80)
    p.add_argument("--min-opponent-stars", type=int, default=40)
    p.add_argument("--shard", type=str, default="0/1")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-concurrent", type=int, default=None)
    a = p.parse_args(argv)
    cmd = [str(gmdata()), "generals", "pipeline", "-p", *a.players, "-o", str(a.output),
           "--target", str(a.target), "--min-turns", str(a.min_turns),
           "--min-opponent-stars", str(a.min_opponent_stars), "--shard", a.shard]
    if a.merge:
        cmd.append("--merge")
    if a.workers:
        cmd += ["--workers", str(a.workers)]
    if a.max_concurrent:
        cmd += ["--max-concurrent", str(a.max_concurrent)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
