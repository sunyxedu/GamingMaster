"""Skat (3 players, the International Skat Server's rules) on the generic
candidate-action adapter:

    rules + frames    data_processing/rs/gmrules/src/games/skat.rs (bidding incl. jump
                      bids, pickup / hand, suit / grand / null, schneider / schwarz /
                      ouvert, card play, ISS scoring; decisions factored to <= 12
                      candidates: bid window + MORE, hand game type + modifier, game +
                      one discard at a time)
    converter         data_processing/rs/gmdata/src/skat.rs (`gmdata skat
                      fetch|verify|stats|convert|render`: the ISS archive, 9.1M games,
                      every move replayed and every recorded result reproduced)
    fetch             scripts/fetch_skat.sh (download → verify → stats → convert --hq)
    simulator         gmsim.Sim("skat") → env/vec_gmsim.py (3 per-seat rows per env)
    model             the generic CandidateEncoder / CandidateDecoder (35 entities:
                      32 cards + 3 seats, 149 globals, 12 candidates; all 0/1 features)
    RL                train_rl_joint's generic CandidateSpec (per-seat streams:
                      ``--actor stream``); reward = ISS's zero-sum money score / 1260
    reference         the ISS rule engine itself (tests/test_skat_reference.py:
                      randomized differential tests, both directions)

Hidden information: a seat sees its own hand, the bids, the skat only after its own
pickup (and its own discards), the declarer's hand in ouvert games and the played
cards — pinned by the causality / privacy tests (tests/test_skat.py).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="skat",
    sim_config={},
    description="Skat, 3 players (ISS rules; IL corpus: the ISS game archive)",
)
