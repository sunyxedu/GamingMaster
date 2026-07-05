"""Othello data pipeline: WTHOR databases → training shards.

Modules:
    rules    exact Othello rules (flips, legality, passes, termination)
    wthor    WTHOR ``.wtb`` binary parser (+ encoder for round-trip tests)
    convert  CLI replaying WTHOR games into ``save_shard`` shards for the
             "othello" adapter (BoardEncoder + PlacementDecoder)
"""
