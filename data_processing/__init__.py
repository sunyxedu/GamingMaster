"""
Python side of the data layer. The pipelines themselves (fetch → filter →
replay/simulate → convert) are Rust: data_processing/rs/gmdata (`gmdata <game> …`)
and data_processing/games/clash_royale/crdata. What stays in Python:

- datasets: ``dataset_seq.SeqReplayDataset`` (generals) and
  ``games.dataset.GameShardDataset`` (gm-shard / .pt shards) for training
- the generals.io reference replay decoder + fog-of-war simulator + gio converter
  (``download.replay_fetcher`` / ``download.deserializer``, ``processing.simulator``,
  ``parsers.generals_json``), used by the replay viewers and ``scripts/play_model.py``;
  the Rust pipeline is tested bit-identical to them
- othello / gomoku rules used by the RL self-play environment

Submodules are imported lazily: the download/filter stages must work on
machines where numpy wheels cannot load (e.g. pre-x86-64-v2 login nodes),
and numpy is only needed by the convert stage.
"""

_LAZY = {
    # Download (no numpy)
    "ReplayFetcher": "data_processing.download.replay_fetcher",
    "download_replays": "data_processing.download.replay_fetcher",
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
