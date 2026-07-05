"""Chess data pipeline: Lichess PGN → shards for the chess adapter.

``convert.py`` streams ``.pgn`` / ``.pgn.zst`` games into the shared shard
format (:mod:`data_processing.games.dataset`); ``fetch_lichess.sh``
downloads monthly Lichess dumps.
"""
