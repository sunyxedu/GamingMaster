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

Full scale (cluster; ~155 GB of shards for both sources, see data_processing/README.md
"bridge")::

    scripts/fetch_bridge.sh ~/gm-data/bridge                    # OpenSpiel + Ben
    SOURCES=openspiel scripts/fetch_bridge.sh ~/gm-data/bridge  # WBridge5 only (~60 GB)
    uv run python -m training.train_multi_game --data bridge=~/gm-data/bridge/train \\
        --val bridge=~/gm-data/bridge/val ...
    uv run python -m training.train_rl_joint --init <multi_final.pt> --rl bridge \\
        --data bridge=~/gm-data/bridge/train --actor stream ...
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="bridge",
    sim_config={},
    description="Contract bridge: auction + card play, duplicate scoring (IMPs), 4 seats / 2 teams",
)
