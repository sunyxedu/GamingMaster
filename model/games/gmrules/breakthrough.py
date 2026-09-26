"""Breakthrough — the CUSTOM-SCHEMA reference game of the gmrules framework: a
perfect-information board game on the board adapter's from-to decoder, the pattern
for chess-like games (chess variants, xiangqi, ...):

    rules + frames    data_processing/rs/gmrules/src/games/breakthrough.rs
                      (schema "fromto.v1": obs board_seq [3, R, C] / globals_seq [2],
                      legal from / to_cond, targets from_idx / to_idx + the IL masks
                      legal_from / legal_to_of), pinned to OpenSpiel
    data              `gmdata rules random|convert --game breakthrough`
    simulator         gmsim.Sim("breakthrough") → env/vec_gmsim.py
    model             model/games/board.py BoardEncoder + FromToDecoder
                      (build_fromto_network; no promotion head)
    RL                train_rl_joint's FromToSpec (teacher-forced from / to replay)

The board dims come from the simulator's spec meta (``sim_config``) or, in a fresh
IL run, from the shard meta (``meta_arch``).
"""

from model.games.gmrules import RulesGame

GAME = RulesGame(
    name="breakthrough",
    sim_config={"rows": 8, "columns": 8},
    description="Breakthrough (custom from-to schema reference game), R x C board",
    builder="model.games.board:build_fromto_network",
    config={"n_promotions": 0},
    label_keys=("from_idx", "to_idx"),
    meta_arch={"board_h": ("board_h",), "board_w": ("board_w",),
               "piece_channels": ("piece_channels",),
               "global_features": ("board_global_features",)},
    sample_keys={"action_from": "from_idx", "action_to": "to_idx"},
    rl_spec="training.train_rl_joint:FromToSpec",
)
