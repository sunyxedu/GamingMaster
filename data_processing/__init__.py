"""
Data processing pipeline for generals.io replays.

Modules:
- download: Fetch replay metadata and files
- parsers: Convert to training format
- processing: Filter and simulate replays

Submodules are imported lazily: the download/filter stages must work on
machines where numpy wheels cannot load (e.g. pre-x86-64-v2 login nodes),
and numpy is only needed by the convert stage.
"""

_LAZY = {
    # Download (no numpy)
    "fetch_replay_list": "data_processing.download.api_client",
    "save_replay_list": "data_processing.download.api_client",
    "ReplayFetcher": "data_processing.download.replay_fetcher",
    "download_replays": "data_processing.download.replay_fetcher",
    # Filter (no numpy)
    "filter_replays": "data_processing.processing.filter",
    "filter_replay_file": "data_processing.processing.filter",
    # Simulate (no numpy)
    "ReplaySimulator": "data_processing.processing.simulator",
    "simulate_replay": "data_processing.processing.simulator",
    # Convert (needs numpy)
    "convert_simulator_to_gio": "data_processing.parsers.generals_json",
    "parse_replay_to_gio": "data_processing.parsers.generals_json",
    "batch_parse_directory": "data_processing.parsers.generals_json",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
