"""Pokémon Showdown replay JSONs → training shards for the ``pokemon`` adapter.

Reads a directory of replay ``.json`` files (as saved by
``fetch_replays.py`` — top-level keys id/format/players/log/rating) and
emits shards in the shared format of :mod:`data_processing.games.dataset`
via :func:`save_shard`.  Files that are not single-replay objects with a
``log`` key (e.g. cached ``search.json`` pages) are skipped.

One sample per (replay, player side); see
:mod:`data_processing.games.pokemon.protocol` for the frame/label spec.
Replays without a ``|win|`` line yield no samples.

CLI
---
    uv run python -m data_processing.games.pokemon.convert \\
        --input replays/ --output_dir shards/ \\
        --shard_size 256 --min_rating 0 --max_replays N

Filters: ``--min_rating`` drops replays whose ``rating`` field is null or
below the threshold (0 = keep everything).  Prints frame counts, a
skip-reason histogram and an unknown-line-prefix histogram at the end.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional

from data_processing.games.dataset import save_shard
from data_processing.games.pokemon.protocol import replay_to_samples


def convert(
    input_dir: str,
    output_dir: str,
    shard_size: int = 256,
    min_rating: int = 0,
    max_replays: Optional[int] = None,
    species_buckets: int = 512,
) -> Dict:
    """Convert every replay JSON under ``input_dir`` into shards.

    Returns counters: seen / kept / samples / frames / shards plus the
    aggregated ``skipped`` and ``unknown`` histograms.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    stats: Dict = {
        "seen": 0, "kept": 0, "samples": 0, "frames": 0, "shards": 0,
        "no_winner": 0, "low_rating": 0, "not_replay": 0,
        "skipped": {}, "unknown": {},
    }
    buf, shard_i = [], 0

    def flush():
        nonlocal buf, shard_i
        if buf:
            save_shard(
                os.path.join(output_dir, f"pokemon_{shard_i:05d}.pt"),
                game="pokemon", samples=buf,
            )
            shard_i += 1
            buf = []

    for path in paths:
        if max_replays is not None and stats["kept"] >= max_replays:
            break
        try:
            with open(path, "r", encoding="utf-8") as fh:
                replay = json.load(fh)
        except (json.JSONDecodeError, OSError):
            stats["not_replay"] += 1
            continue
        if not isinstance(replay, dict) or "log" not in replay:
            stats["not_replay"] += 1     # e.g. a search.json page
            continue
        stats["seen"] += 1
        if min_rating > 0:
            rating = replay.get("rating") or 0
            if rating < min_rating:
                stats["low_rating"] += 1
                continue

        samples, result = replay_to_samples(replay, species_buckets=species_buckets)
        for k, v in result.skipped.items():
            stats["skipped"][k] = stats["skipped"].get(k, 0) + v
        for k, v in result.unknown.items():
            stats["unknown"][k] = stats["unknown"].get(k, 0) + v
        if result.winner is None:
            stats["no_winner"] += 1
            continue
        if not samples:
            continue
        stats["kept"] += 1
        for s in samples:
            stats["samples"] += 1
            stats["frames"] += int(s["tgt"]["action_type"].shape[0])
            buf.append(s)
            if len(buf) >= shard_size:
                flush()
    flush()
    stats["shards"] = shard_i
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="directory of replay .json files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=256)
    p.add_argument("--min_rating", type=int, default=0,
                   help=">0 also drops replays with a null rating")
    p.add_argument("--max_replays", type=int, default=None,
                   help="stop after keeping this many replays")
    p.add_argument("--species_buckets", type=int, default=512,
                   help="must match the model's pokemon_species_buckets")
    args = p.parse_args()

    stats = convert(
        args.input, args.output_dir,
        shard_size=args.shard_size, min_rating=args.min_rating,
        max_replays=args.max_replays, species_buckets=args.species_buckets,
    )
    print(f"[pokemon] seen={stats['seen']} kept={stats['kept']} "
          f"samples={stats['samples']} frames={stats['frames']} "
          f"shards={stats['shards']} no_winner={stats['no_winner']} "
          f"low_rating={stats['low_rating']} not_replay={stats['not_replay']} "
          f"→ {args.output_dir}")
    if stats["skipped"]:
        print("[pokemon] skipped turns: " + " ".join(
            f"{k}={v}" for k, v in sorted(stats["skipped"].items())))
    if stats["unknown"]:
        print("[pokemon] unknown line prefixes: " + " ".join(
            f"{k}={v}" for k, v in sorted(stats["unknown"].items(),
                                          key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
