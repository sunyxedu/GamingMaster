"""Reference dumps of the International Skat Server's own rule engine for the
differential tests of the gmrules skat rules (tests/test_skat_reference.py):

    python scripts/reference/iss_skat_dump.py --play 200 --seed 1 --out ref.jsonl
    # replay ISS record lines (archive format; e.g. `gmdata skat render` of gmrules
    # trajectories, or real archive records) through the ISS engine
    python scripts/reference/iss_skat_dump.py --replay games.sgf --out ref.jsonl

Every game records, per decision, the seat, the ISS phase, the max bid, the moves the
ISS engine accepts (probed black-box on copies of its state) and the move taken, then
the engine's result string (`GameResult`) and money scores — see IssSkatRef.java.

Needs a JDK (javac + java, 8 or newer) and, the first time, network: the ISS client
package (https://skatgame.net/iss/clients.tgz, 3.1 MB, SHA-256 pinned; (c) Michael
Buro, Jeff Long, Jan Schaefer, Nathan Taylor, GPLv3 — the server's rule classes
SimpleState / GameDeclaration / GameResult / Card / Hand) is downloaded into
$GM_REFERENCE_CACHE (default ~/.cache/gamingmaster/reference) and compiled there with
IssSkatRef.java. Nothing of it enters the repository; stdlib only (runs under any
Python 3.8+).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://skatgame.net/iss/clients.tgz"
SHA256 = "302fe9d4b6a5c590c3f1a3028dae1021f98031c28e66ca6a4eb9f5e827723656"
#: the rule classes IssSkatRef needs (javac -sourcepath pulls in the rest of common/)
SOURCES = ["common/SimpleState.java", "common/GameResult.java", "common/GameDeclaration.java"]
DRIVER = Path(__file__).resolve().parent / "IssSkatRef.java"


def cache_dir() -> Path:
    default = Path.home() / ".cache" / "gamingmaster" / "reference"
    return Path(os.environ.get("GM_REFERENCE_CACHE", default)) / "iss-skat"


def have_jdk() -> bool:
    return shutil.which("javac") is not None and shutil.which("java") is not None


def fetch(root: Path) -> Path:
    """The extracted client sources (downloaded and checked once)."""
    src = root / "skatgui" / "src"
    if (src / SOURCES[0]).exists():
        return src
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "clients.tgz"
    if not tgz.exists():
        tmp = tgz.with_suffix(".part")
        req = urllib.request.Request(URL, headers={"User-Agent": "GamingMaster-research/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.rename(tgz)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if digest != SHA256:
        tgz.unlink()
        raise SystemExit(f"{URL}: sha256 {digest}, expected {SHA256} (package changed?)")
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith("skatgui/src/common/") and m.isfile()]
        tf.extractall(root, members=members)
    return src


def build(root: Path) -> Path:
    """Compiled classes of the ISS rule engine + the driver (rebuilt when the driver changes)."""
    src = fetch(root)
    classes = root / "classes"
    stamp = classes / ".driver.sha256"
    want = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    if stamp.exists() and stamp.read_text() == want:
        return classes
    if classes.exists():
        shutil.rmtree(classes)
    classes.mkdir(parents=True)
    cmd = ["javac", "-nowarn", "-encoding", "ISO-8859-1", "-d", str(classes),
           "-sourcepath", str(src), str(DRIVER)] + [str(src / s) for s in SOURCES]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"javac failed:\n{r.stderr[-3000:]}")
    stamp.write_text(want)
    return classes


def run(args, out: Path, root: Path | None = None) -> None:
    classes = build(root or cache_dir())
    with open(out, "w") as fh:
        subprocess.run(["java", "-cp", str(classes), "IssSkatRef", *args], stdout=fh, check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--play", type=int, metavar="GAMES", help="random playouts of the ISS engine")
    g.add_argument("--replay", type=Path, metavar="FILE", help="ISS record lines to replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not have_jdk():
        raise SystemExit("needs a JDK (javac and java on PATH)")
    if a.play is not None:
        run(["play", str(a.play), str(a.seed)], a.out)
    else:
        run(["replay", str(a.replay.resolve())], a.out)


if __name__ == "__main__":
    sys.exit(main())
