"""Dou Dizhu (斗地主) — three players, an auction for the landlord (who takes the
three kitty cards, shown to everyone) against two peasants; the first to empty their
hand wins for their side.  Standard rules = OpenSpiel's ``dou_dizhu`` (auction bids
1-3, every combination: solos, pairs, trios with a solo / a pair, chains, pair
chains, airplanes with or without wings, bombs, the rocket) + four-with-two;
score = bid x 2^(bombs + rockets + spring), the landlord wins / loses twice a
peasant's share.  Every layer:

    rules + frames    data_processing/rs/gmrules/src/games/doudizhu/ (the candidate
                      schema, the acting seat's view only: own hand, public plays,
                      card counts, the auction, the face-up card, the kitty once shown)
    parity            OpenSpiel dou_dizhu (tests/test_doudizhu_reference.py: rules="openspiel"
                      identical step for step, both directions) and DouZero's play-phase
                      engine (tests/test_doudizhu_douzero.py)
    data              scripts/generate_doudizhu.py (the public DouZero ADP / WP agents play
                      every card, a documented heuristic bids: there is no public human
                      data) → gmdata doudizhu convert (replays and verifies every record)
                      — end to end: scripts/fetch_doudizhu.sh
    simulator         gmsim.Sim("doudizhu") → env/vec_gmsim.py (3 per-seat rows per env,
                      alternating: one active row per step; the RL env includes the auction,
                      so bidding is learned by RL)
    model             the generic CandidateEncoder / CandidateDecoder (this entry)
    RL                train_rl_joint's generic CandidateSpec (--actor stream)

Decisions.  A combination with kickers is chosen in consecutive decisions of the
same seat: its main part, then one kicker (solo / pair) per decision in rank order,
only completable prefixes offered — so a frame never has more than 92 candidates
(exact: the most main-level options of any 20-card hand, 91, + pass; the auction has
at most 4, a kicker decision at most 15), while a hand may hold up to 521 full plays.
Frame: 20 entities x 43 features (15 rank slots — my cards, unseen cards, cards
played by each seat, the kitty, the play to beat, the last plays, the combination
under construction — 3 seat slots, the play to beat, the pending combination),
24 globals, 92 candidates x 48 features, all 0/1 (bit-packed shards, ~0.8 KB per
frame).  Outcome (IL value target, RL reward): returns / 24 clipped to [-1, 1]
(``scoring="wp"``: ±1 per side).

Corpus (``scripts/fetch_doudizhu.sh``; DouZero self-play, ~36 frames and ~28 KB of
shards per game on average, 3 per-seat samples): 1M games ≈ 36M frames ≈ 28 GB,
generated at ~22 games/s per CPU worker (+ 20k validation games).  Full scale on the
cluster (from the repository root)::

    # login node (internet): DouZero sources + ADP / WP checkpoints → ~/.cache/gm-reference
    .venv/bin/python scripts/generate_doudizhu.py --fetch_only
    # a 64-core compute node: generate 1M + 20k games, convert both with --hq
    WORKERS=64 PYTHON=.venv/bin/python scripts/fetch_doudizhu.sh data/doudizhu 1000000
    # → data/doudizhu/train (~36M frames, ~28 GB), data/doudizhu/val (~0.7M frames)

The auction frames of the corpus imitate the heuristic (``meta.bidding =
"heuristic_v1"``); the RL phase plays the real auction.
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="doudizhu",
    sim_config={},
    description="Dou Dizhu, 3 players: auction, landlord vs two peasants (DouZero self-play IL)",
)
