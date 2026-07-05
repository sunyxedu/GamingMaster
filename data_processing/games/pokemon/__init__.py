"""Pokémon Showdown data pipeline: replay JSON → training shards for the
``pokemon`` adapter.

Modules:
    protocol      — sim-protocol log parser → per-turn observer-view states
    convert       — replay JSON dir → shard converter + CLI
    fetch_replays — replay.pokemonshowdown.com search.json downloader
"""
