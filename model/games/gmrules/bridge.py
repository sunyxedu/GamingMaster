"""Contract bridge (auction + card play, duplicate scoring) — a candidate-action
gmrules game, 4 seats in two partnerships:

    rules + frames    data_processing/rs/gmrules/src/games/bridge.rs (OpenSpiel's
                      bridge exactly: tests/test_bridge_reference.py)
    converter         data_processing/rs/gmdata/src/bridge.rs — `gmdata bridge convert
                      --format openspiel|jack|pbn|lin`, every record replayed and
                      verified (contract and result included where recorded)
    fetch             scripts/fetch_bridge.sh (OpenSpiel WBridge5 dataset + Ben's WBC
                      play data → verified shards)
    simulator         gmsim.Sim("bridge") → env/vec_gmsim.py (per-seat streams; the
                      declarer's row carries dummy's plays, to_play = declarer as in
                      OpenSpiel; ``teams`` = N/S vs E/W, the league gives a policy a whole
                      partnership)
    model             the generic CandidateEncoder / CandidateDecoder: 91 entity
                      tokens (52 cards, 35 bids, 4 seats — slot = identity), the global
                      token, up to 38 candidate tokens (the legal calls or cards)
    RL                train_rl_joint's generic CandidateSpec (``--actor stream``)

Hidden information: a frame shows its seat's own hand, dummy's after the opening lead,
and the public auction / play; dummy never decides in the play (its plays are
declarer's frames).  The value target / reward is the IMPs of the seat's side's score
against 0, / 24.  The simulator draws the board (dealer + vulnerability) per deal;
records carry theirs, so one network plays every board.

Full scale (cluster; Mac + Linux, needs cargo — the script builds gmdata)::

    scripts/fetch_bridge.sh $HOME/gm-data/bridge                    # both sources
    SOURCES=openspiel scripts/fetch_bridge.sh $HOME/gm-data/bridge  # WBridge5 only
    MAX_RECORDS=300 scripts/fetch_bridge.sh /tmp/bridge_smoke       # capped smoke run

Expected corpus (downloads 0.38 GB + 0.15 GB; ~810 bytes per decision frame):
OpenSpiel WBridge5 train 1,158,441 deals / 72.7M frames / ~59 GB, valid and test
10,000 deals each (~0.63M frames, ~0.5 GB); Ben WBC 1,883,853 of 1,887,139 boards
kept (the rest are data errors the verifying replay rejects), 117.9M frames / ~95 GB,
1 in 100 deals routed to val by a hash of the deal.  Layout: OUT/train/{openspiel,ben},
OUT/val/{openspiel,ben}, OUT/test/openspiel — one shard family, so one root per split::

    uv run python -m training.train_multi_game --data bridge=$HOME/gm-data/bridge/train \\
        --val bridge=$HOME/gm-data/bridge/val --game_frame_cap bridge=1e8 ...
    uv run python -m training.train_rl_joint --init <multi_final.pt> --rl bridge \\
        --data bridge=$HOME/gm-data/bridge/train ...
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="bridge",
    sim_config={},
    description="Contract bridge: auction + card play, duplicate scoring (IMPs), 4 seats / 2 teams",
)
