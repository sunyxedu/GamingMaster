"""Xiangqi (Chinese chess) — a from-to board game of the gmrules framework, on the
board adapter (the pattern of model/games/gmrules/breakthrough.py):

    rules + frames    data_processing/rs/gmrules/src/games/xiangqi/ — Asian rules as
                      Fairy-Stockfish implements them (flying general, cannon screens,
                      AXF perpetual check / chase, 50-move limit with the check
                      extension); schema "fromto.v1", every frame from the mover's
                      side (Black's board rotated by 180°): obs board_seq [19, 10, 9]
                      (own / opponent K A B N R C P, last move, AXF chase planes) /
                      globals_seq [7] (mover, clocks, check, repetition, check runs),
                      legal from [90] / to_cond [90, 90], targets from_idx / to_idx +
                      the IL masks legal_from / legal_to_of; pinned to Fairy-Stockfish
                      (tests/test_xiangqi_reference.py)
    data              `gmdata xiangqi fetch|convert` (CCPD master games, CC BY 4.0;
                      scripts/fetch_xiangqi.sh), `gmdata rules random --game xiangqi`
    simulator         gmsim.Sim("xiangqi") → env/vec_gmsim.py (interleaved: one row
                      per game, lockstep)
    model             model/games/board.py BoardEncoder + FromToDecoder
                      (build_fromto_network; no promotion head)
    RL                train_rl_joint's FromToSpec (teacher-forced from / to replay)

The board dims come from the simulator's spec meta or, in a fresh IL run, from the
shard meta (``meta_arch``).  Pure policy play: ``GameNetwork.step`` + the env's exact
legality tables (``from`` / ``to_cond``) — every sampled (from, to) is a legal move.
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="xiangqi",
    sim_config={},
    description="Xiangqi (Chinese chess), Asian rules (Fairy-Stockfish), from-to board schema",
    builder="model.games.board:build_fromto_network",
    config={"n_promotions": 0},
    label_keys=("from_idx", "to_idx"),
    meta_arch={"board_h": ("board_h",), "board_w": ("board_w",),
               "piece_channels": ("piece_channels",),
               "global_features": ("board_global_features",)},
    sample_keys={"action_from": "from_idx", "action_to": "to_idx"},
    rl_spec="training.train_rl_joint:FromToSpec",
)
