"""Kuhn poker (2-4 players) — the toy reference game of the candidate-action
framework, implemented through every layer:

    rules + frames    data_processing/rs/gmrules/src/games/kuhn_poker.rs
    converter         data_processing/rs/gmdata/src/kuhn_poker.rs (hand logs → shards)
    fetch             scripts/fetch_kuhn_poker.sh (synthetic logs → shards)
    simulator         gmsim.Sim("kuhn_poker") → env/vec_candidate.py
    model             the generic CandidateEncoder / CandidateDecoder
    RL                train_rl_joint's generic CandidateSpec

Frame dims (5 entities, 2 candidates, ...) come from the simulator's FrameSpec;
they are the same for 2, 3 and 4 players, so one network plays every variant.
"""

from model.games.candidates import CandidateGame

GAME = CandidateGame(
    name="kuhn_poker",
    sim_config={"players": 2},
    description="Kuhn poker, 2-4 players (toy reference game)",
)
