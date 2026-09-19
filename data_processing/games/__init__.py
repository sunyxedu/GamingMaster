"""Multi-game data pipelines.

Each game package converts its native replay format (PGN / SGF / WTHOR /
CSA / …) into the shared shard format consumed by
:mod:`data_processing.games.dataset` and trained by ``training/train_il_game.py``.
"""
