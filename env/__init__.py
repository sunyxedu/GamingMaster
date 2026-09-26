"""Self-play environments: the interactive generals.io env, the board-game
self-play env, and the vectorised VecGameEnv adapters for mixed multi-game
RL (env/vec_game.py; seat / opponent control in env/league.py)."""

from .generals_env import GeneralsEnv
from .vec_game import VecGameEnv, VecObs, has_simulator, make_vec_env

__all__ = ["GeneralsEnv", "VecGameEnv", "VecObs", "has_simulator", "make_vec_env"]
