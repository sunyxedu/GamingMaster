"""Goofspiel — the simultaneous-move test game of the candidate-action framework
(data_processing/rs/gmrules/src/games/goofspiel.rs): per-seat streams, every seat
acting on every turn.  No converter; random-play shards come from
``gmdata rules random --game goofspiel``.
"""

from model.games.candidates import CandidateGame

GAME = CandidateGame(
    name="goofspiel",
    sim_config={"num_cards": 5, "players": 2},
    description="Goofspiel (simultaneous bids), 2-4 players, 3-13 cards",
)
