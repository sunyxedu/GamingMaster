"""Write the start-up sidecars (index.head.json + index.lengths.npy) of every gm-shard under
the given roots, so GameShardDataset opens a corpus without parsing each shard's full
index.json (the HUNL corpus: 13,820 shards x ~4.4 MB of per-sample JSON per process).

    uv run python scripts/build_shard_index.py ~/gm-data/hunl3 [ROOT ...] [--workers 8] [--force]

Idempotent: a shard whose sidecars are at least as new as its index.json is skipped (unless
--force).  The writes are atomic, so a training job reading the root meanwhile sees either no
sidecar (and parses index.json) or a complete one.  Re-run after a shard is re-converted —
a stale sidecar (older than index.json) is ignored by the loader anyway."""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.games.dataset import (HEAD_SIDECAR, LENGTHS_SIDECAR,  # noqa: E402
                                           find_gm_shards, write_shard_sidecars)


def fresh(path: str) -> bool:
    try:
        t = os.stat(os.path.join(path, "index.json")).st_mtime
        return all(os.stat(os.path.join(path, f)).st_mtime >= t
                   for f in (HEAD_SIDECAR, LENGTHS_SIDECAR))
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    shards = sorted({os.path.realpath(p) for r in a.roots for p in find_gm_shards(r)})
    todo = shards if a.force else [p for p in shards if not fresh(p)]
    print(f"{len(shards)} shards, {len(todo)} to index ({time.time() - t0:.0f} s to list)", flush=True)
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, k in enumerate(ex.map(write_shard_sidecars, todo, chunksize=16)):
            n += k
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)} shards, {n:,} samples, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {len(todo)} shards, {n:,} samples in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
