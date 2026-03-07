"""
Data processing pipeline for generals.io replays.

Modules:
- download: Fetch replay metadata and files
- processing: Filter and simulate replays
- parsers: Convert to training format
"""

from data_processing.download.api_client import fetch_replay_list, save_replay_list
from data_processing.download.replay_fetcher import ReplayFetcher, download_replays
from data_processing.parsers.generals_json import (
    batch_parse_directory,
    convert_simulator_to_gio,
    parse_replay_to_gio,
)
from data_processing.processing.filter import filter_replay_file, filter_replays
from data_processing.processing.simulator import ReplaySimulator, simulate_replay

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