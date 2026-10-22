"""Write the data-root manifests (data_processing/games/dataset.py, MANIFEST) of gm-shard roots:
every shard's header and frame counts in ``<root>/.gm_manifest.npz``, so a dataset over the root
opens from one file instead of a walk into and two reads from every shard.

    uv run python scripts/build_data_manifest.py ROOT [ROOT ...]          # stale / missing only
    uv run python scripts/build_data_manifest.py --force ROOT [ROOT ...]  # rebuild all

A manifest is trusted while the root's non-shard directories list the same entries; rebuild it
(--force) after re-converting a shard in place.  Roots holding .pt shards are skipped."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.games.dataset import (MANIFEST, GameShardDataset, load_data_manifest,  # noqa: E402
                                           write_data_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--force", action="store_true", help="rebuild fresh manifests too")
    ap.add_argument("--check", action="store_true",
                    help="after writing, open the root both ways and compare (slow: walks it)")
    a = ap.parse_args()
    bad = 0
    for root in a.roots:
        root = os.path.expanduser(root.rstrip("/"))
        if not os.path.isdir(root):
            print(f"{root}: not a directory", flush=True)
            bad += 1
            continue
        if not a.force and load_data_manifest(root) is not None:
            print(f"{root}: manifest fresh", flush=True)
            continue
        t = time.time()
        out = write_data_manifest(root)
        if out is None:
            print(f"{root}: skipped (.pt shards, a single shard, or none)", flush=True)
            continue
        man = load_data_manifest(root)
        n = len(man[0]) if man else 0
        print(f"{root}: {n} shards -> {MANIFEST} in {time.time() - t:.0f}s", flush=True)
        if man is None:
            print(f"{root}: the manifest just written does not load", flush=True)
            bad += 1
        elif a.check:
            os.environ["GM_DATA_MANIFEST"] = "0"
            scan = GameShardDataset(root)
            os.environ.pop("GM_DATA_MANIFEST")
            fast = GameShardDataset(root)
            ok = (fast.from_manifest and scan.paths == fast.paths
                  and scan.sample_lengths().tolist() == fast.sample_lengths().tolist()
                  and scan.frame_stats() == fast.frame_stats())
            print(f"{root}: check {'ok' if ok else 'MISMATCH'}", flush=True)
            bad += 0 if ok else 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
