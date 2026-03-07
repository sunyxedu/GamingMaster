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
    download  - Fetch replay metadata and download replay files
    filter    - Filter replays by criteria (1v1, min turns, etc.)
    simulate  - Simulate replays from player perspective with fog of war
    convert   - Convert simulated data to training format
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from data_processing import (
    fetch_replay_list,
    download_replays,
    filter_replays,
    simulate_replay,
    batch_parse_directory,
)


ALL_STAGES = ["download", "filter", "simulate", "convert"]


async def run_download(
    players: List[str],
    output_dir: Path,
    max_replays: int,
    max_concurrent: int,
) -> dict[str, List[dict]]:
    """Download replay metadata and files for given players."""
    metadata_dir = output_dir / "metadata"
    replays_dir = output_dir / "replays"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    replays_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for player in players:
        print(f"\n=== Downloading replays for {player} ===")

        # Fetch metadata
        metadata_file = metadata_dir / f"{player}.json"
        if metadata_file.exists():
            print(f"Loading existing metadata from {metadata_file}")
            with open(metadata_file) as f:
                replays = json.load(f)
        else:
            print(f"Fetching replay list for {player}...")
            replays = await fetch_replay_list(player, total=max_replays)
            with open(metadata_file, "w") as f:
                json.dump(replays, f, indent=2)
            print(f"Saved {len(replays)} replay metadata to {metadata_file}")

        all_metadata[player] = replays

        # Download replay files
        replay_ids = [r["id"] for r in replays]
        print(f"Downloading {len(replay_ids)} replay files...")
        result = await download_replays(
            replay_ids, str(replays_dir), max_concurrent=max_concurrent
        )
        print(f"Downloaded: {result['success']} success, {result['failed']} failed")

    return all_metadata


def run_filter(
    players: List[str],
    output_dir: Path,
    min_turns: int,
    min_opponent_stars: Optional[int],
) -> dict[str, List[dict]]:
    """Filter replays based on criteria."""
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
        )

        filtered_file = filtered_dir / f"{player}.json"
        with open(filtered_file, "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"Output: {len(filtered)} replays (saved to {filtered_file})")
        all_filtered[player] = filtered

    return all_filtered


def _simulate_one(args):
    """Worker function for parallel simulation (must be top-level for pickling)."""
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
        choices=ALL_STAGES,
        default=ALL_STAGES,
        help=f"Stages to run (default: all). Choices: {ALL_STAGES}",
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=10000,
        help="Maximum replays to fetch per player (default: 10000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent downloads (default: 10)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=80,
        help="Minimum turns required (default: 80)",
    )
    parser.add_argument(
        "--min-opponent-stars",
        type=int,
        default=None,
        help="Minimum opponent stars required (optional)",
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