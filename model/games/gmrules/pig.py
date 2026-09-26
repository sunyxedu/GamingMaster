"""Pig — the perfect-information dice test game of the candidate-action framework
(data_processing/rs/gmrules/src/games/pig.rs): the INTERLEAVED layout (one stream
per game, every decision from the mover's view — the backgammon-like case), chance
events, 2-4 players.  No converter; random-play shards come from
``gmdata rules random --game pig``.
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="pig",
    sim_config={"players": 2, "winscore": 20, "diceoutcomes": 6, "horizon": 200},
    description="Pig (roll / stop dice game), 2-4 players",
)
