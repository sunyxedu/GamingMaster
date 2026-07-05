"""Go data pipeline: SGF records → training shards for the ``go`` adapter.

Modules:
    sgf     — minimal main-line-only SGF parser (pure python)
    rules   — Go rules engine (captures, suicide, simple ko)
    convert — SGF → shard converter + CLI
"""
