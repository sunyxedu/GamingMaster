#!/usr/bin/env python3
"""
Data preparation script for generals.io replays.

Usage:
    # Run all stages
    uv run python scripts/prepare_data.py --players MeltedToast --output data/output

    # Run specific stages
    uv run python scripts/prepare_data.py --players MeltedToast --output data/output --stages download filter

    # Run only convert stage on existing data
    uv run python scripts/prepare_data.py --players MeltedToast --output data/output --stages convert

Stages:
    download  - Collect --max-replays TRAINABLE games: fetch metadata
                page-by-page, filter, download candidate files and SIMULATE
                them; only games the simulator accepts count (map size,
                modifiers and modern features are only visible in the file)
    filter    - (optional) re-filter cached metadata offline
    simulate  - (usually a no-op after download) simulate leftover games
    convert   - Convert simulated data to training format (needs numpy)

--max-replays counts games that survive filter AND simulation, i.e. games
that actually enter the training set.
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# Stage imports happen inside each run_* function: numpy is only needed by
# the convert stage, and download/filter must work on machines where numpy
# wheels cannot load (e.g. pre-x86-64-v2 login nodes).


# "download" fetches+filters incrementally and downloads filtered games only;
# "filter" remains available for re-filtering cached metadata offline.
ALL_STAGES = ["download", "simulate", "convert"]
STAGE_CHOICES = ["download", "filter", "simulate", "convert"]


FILTER_START_DATE = "2025-04-01T00:00:00Z"


async def run_download(
    players: List[str],
    output_dir: Path,
    max_replays: int,
    max_concurrent: int,
    min_turns: int,
    min_opponent_stars: Optional[int],
    sim_workers: Optional[int] = None,
) -> dict[str, List[dict]]:
    """Collect ``max_replays`` TRAINABLE games per player.

    Loop: fetch metadata page-by-page → filter → download candidate files →
    SIMULATE them. Only games the simulator accepts count toward the target
    (the metadata filter cannot see map size / modifiers / modern features —
    those live inside the replay file and are rejected at simulation).
    Rejected ids go to ``simulated/<player>_skipped.json`` so reruns don't
    redo them; ``filtered/<player>.json`` ends up listing exactly the games
    that entered ``simulated/`` (and hence training).
    """
    from data_processing import download_replays, filter_replays
    from data_processing.download.api_client import fetch_replay_pages

    metadata_dir = output_dir / "metadata"
    filtered_dir = output_dir / "filtered"
    replays_dir = output_dir / "replays"
    simulated_dir = output_dir / "simulated"
    for d in (metadata_dir, filtered_dir, replays_dir, simulated_dir):
        d.mkdir(parents=True, exist_ok=True)

    CHUNK = 16  # candidates downloaded+simulated per round

    def passes(entries: List[dict]) -> List[dict]:
        return filter_replays(
            entries,
            min_turns=min_turns,
            player_name=player,
            min_opponent_stars=min_opponent_stars,
            start_date=FILTER_START_DATE,
        )

    all_filtered = {}
    for player in players:
        print(f"\n=== Collecting replays for {player} "
              f"(target: {max_replays} trainable) ===")

        sim_dir = simulated_dir / player
        sim_dir.mkdir(parents=True, exist_ok=True)
        skip_file = simulated_dir / f"{player}_skipped.json"
        skipped: dict = json.load(open(skip_file)) if skip_file.exists() else {}

        # Resume: cached metadata (newest-first, append-only) + existing sims
        metadata_file = metadata_dir / f"{player}.json"
        raw: List[dict] = []
        if metadata_file.exists():
            raw = json.load(open(metadata_file))
            print(f"Resuming from {len(raw)} cached metadata entries, "
                  f"{len(skipped)} known-skipped")

        ok_entries: List[dict] = []
        pending: List[dict] = []
        for e in passes(raw):
            if (sim_dir / f"{e['id']}.json").exists():
                ok_entries.append(e)
            elif e["id"] not in skipped:
                pending.append(e)

        pages = fetch_replay_pages(player, start_offset=len(raw))
        exhausted = False

        with ProcessPoolExecutor(max_workers=sim_workers) as pool:
            while len(ok_entries) < max_replays:
                # Refill the candidate queue from the API
                while not pending and not exhausted:
                    try:
                        page = await anext(pages)
                    except StopAsyncIteration:
                        exhausted = True
                        break
                    raw.extend(page)
                    pending.extend(
                        e for e in passes(page)
                        if e["id"] not in skipped
                        and not (sim_dir / f"{e['id']}.json").exists()
                    )
                    print(f"  fetched {len(raw)} games, "
                          f"candidates +{len(pending)}")
                if not pending:
                    print(f"  history exhausted: only {len(ok_entries)} "
                          f"trainable games")
                    break

                take = min(CHUNK, max_replays - len(ok_entries))
                batch, pending = pending[:take], pending[take:]
                ids = [e["id"] for e in batch]
                await download_replays(ids, str(replays_dir),
                                       max_concurrent=max_concurrent)

                futures = {}
                for e in batch:
                    replay_file = replays_dir / f"{e['id']}.json"
                    if not replay_file.exists():
                        print(f"  download failed: {e['id']} (will retry on rerun)")
                        continue
                    out_file = sim_dir / f"{e['id']}.json"
                    futures[pool.submit(
                        _simulate_one, (str(replay_file), player, str(out_file))
                    )] = e
                for fut in as_completed(futures):
                    e = futures[fut]
                    try:
                        fut.result()
                        ok_entries.append(e)
                    except ValueError as err:  # simulator rejection — permanent
                        skipped[e["id"]] = str(err)
                        print(f"  skipped {e['id']}: {err}")
                    except Exception as err:  # transient — retry on rerun
                        print(f"  error simulating {e['id']}: {err}")
                print(f"  trainable {len(ok_entries)}/{max_replays}")

        with open(metadata_file, "w") as f:
            json.dump(raw, f, indent=2)
        with open(skip_file, "w") as f:
            json.dump(skipped, f, indent=2)
        filtered_file = filtered_dir / f"{player}.json"
        with open(filtered_file, "w") as f:
            json.dump(ok_entries, f, indent=2)
        print(f"Collected {len(ok_entries)} trainable games "
              f"(saved to {filtered_file})")
        all_filtered[player] = ok_entries

    return all_filtered


def run_filter(
    players: List[str],
    output_dir: Path,
    min_turns: int,
    min_opponent_stars: Optional[int],
) -> dict[str, List[dict]]:
    """Filter replays based on criteria."""
    from data_processing import filter_replays

    metadata_dir = output_dir / "metadata"
    filtered_dir = output_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    all_filtered = {}

    for player in players:
        print(f"\n=== Filtering replays for {player} ===")

        metadata_file = metadata_dir / f"{player}.json"
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            print("Run 'download' stage first.")
            continue

        with open(metadata_file) as f:
            replays = json.load(f)

        print(f"Input: {len(replays)} replays")

        filtered = filter_replays(
            replays,
            min_turns=min_turns,
            player_name=player,
            min_opponent_stars=min_opponent_stars,
            start_date=FILTER_START_DATE,
        )

        filtered_file = filtered_dir / f"{player}.json"
        with open(filtered_file, "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"Output: {len(filtered)} replays (saved to {filtered_file})")
        all_filtered[player] = filtered

    return all_filtered


def _simulate_one(args):
    """Worker function for parallel simulation (must be top-level for pickling)."""
    from data_processing import simulate_replay

    replay_file, player, output_file = args
    simulate_replay(replay_file, player, output_file)
    return replay_file


def run_simulate(players: List[str], output_dir: Path, max_workers: int = None) -> int:
    """Simulate replays from each player's perspective using multiprocessing."""
    filtered_dir = output_dir / "filtered"
    replays_dir = output_dir / "replays"
    simulated_dir = output_dir / "simulated"

    total_simulated = 0

    for player in players:
        print(f"\n=== Simulating replays for {player} ===")

        player_sim_dir = simulated_dir / player
        player_sim_dir.mkdir(parents=True, exist_ok=True)

        filtered_file = filtered_dir / f"{player}.json"
        if not filtered_file.exists():
            print(f"Filtered file not found: {filtered_file}")
            print("Run 'filter' stage first.")
            continue

        with open(filtered_file) as f:
            filtered = json.load(f)

        # Collect tasks to submit
        tasks = []
        already_done = 0
        for replay in filtered:
            replay_id = replay["id"]
            replay_file = replays_dir / f"{replay_id}.json"
            output_file = player_sim_dir / f"{replay_id}.json"

            if output_file.exists():
                already_done += 1
                continue

            if not replay_file.exists():
                print(f"Replay file not found: {replay_file}")
                continue

            tasks.append((str(replay_file), player, str(output_file)))

        simulated_count = already_done
        if tasks:
            print(f"Simulating {len(tasks)} replays concurrently ({already_done} already done)...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_simulate_one, t): t for t in tasks}
                for future in as_completed(futures):
                    replay_file, _, _ = futures[future]
                    replay_id = Path(replay_file).stem
                    try:
                        future.result()
                        simulated_count += 1
                    except ValueError as e:
                        print(f"Skipping {replay_id}: {e}")
                    except Exception as e:
                        print(f"Error simulating {replay_id}: {e}")

        print(f"Simulated {simulated_count} replays for {player}")
        total_simulated += simulated_count

    return total_simulated


def run_convert(players: List[str], output_dir: Path) -> int:
    """Convert simulated data to training format."""
    from data_processing import batch_parse_directory

    simulated_dir = output_dir / "simulated"
    training_dir = output_dir / "training"

    total_converted = 0

    for player in players:
        print(f"\n=== Converting data for {player} ===")

        player_sim_dir = simulated_dir / player
        player_train_dir = training_dir / player

        if not player_sim_dir.exists():
            print(f"Simulated directory not found: {player_sim_dir}")
            print("Run 'simulate' stage first.")
            continue

        count = batch_parse_directory(str(player_sim_dir), str(player_train_dir))
        total_converted += count

    return total_converted


async def async_main():
    parser = argparse.ArgumentParser(
        description="Prepare generals.io replay data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--players",
        "-p",
        nargs="+",
        required=True,
        help="Player usernames to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--stages",
        "-s",
        nargs="+",
        choices=STAGE_CHOICES,
        default=ALL_STAGES,
        help=f"Stages to run (default: all). Choices: {ALL_STAGES}",
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=10000,
        help="Target number of TRAINABLE replays per player (default: 10000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent downloads (default: os.cpu_count())",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=80,
        help="Minimum turns required (default: 80)",
    )
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=None,
        help="Parallel simulation workers in the download loop (default: CPU count)",
    )
    parser.add_argument(
        "--min-opponent-stars",
        type=int,
        default=40,
        help="Minimum opponent stars required (default: 60)",
    )

    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Players: {args.players}")
    print(f"Output: {output_dir}")
    print(f"Stages: {args.stages}")
    print("=" * 50)

    if "download" in args.stages:
        await run_download(
            args.players,
            output_dir,
            args.max_replays,
            args.max_concurrent,
            args.min_turns,
            args.min_opponent_stars,
            sim_workers=args.sim_workers,
        )

    if "filter" in args.stages:
        run_filter(
            args.players,
            output_dir,
            args.min_turns,
            args.min_opponent_stars,
        )

    if "simulate" in args.stages:
        run_simulate(args.players, output_dir)

    if "convert" in args.stages:
        run_convert(args.players, output_dir)

    print("\n" + "=" * 50)
    print("Done!")


def main():
    """Entry point for the CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()