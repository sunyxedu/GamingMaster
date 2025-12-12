"""
Data processing pipeline for generals.io replays.

Modules:
- download: Fetch replay metadata and files
- processing: Filter and simulate replays
- parsers: Convert to training format
"""

from data_processing.download.api_client import fetch_replay_list, save_replay_list
from data_processing.download.replay_fetcher import ReplayFetcher, download_replays
from data_processing.processing.filter import filter_replays, filter_replay_file
from data_processing.processing.simulator import ReplaySimulator, simulate_replay
from data_processing.parsers.generals_json import (
    convert_simulator_to_gio,
    parse_replay_to_gio,
    batch_parse_directory,
)

__all__ = [
    # Download
    "fetch_replay_list",
    "save_replay_list",
    "ReplayFetcher",
    "download_replays",
    # Filter
    "filter_replays",
    "filter_replay_file",
    # Simulate
    "ReplaySimulator",
    "simulate_replay",
    # Convert
    "convert_simulator_to_gio",
    "parse_replay_to_gio",
    "batch_parse_directory",
]